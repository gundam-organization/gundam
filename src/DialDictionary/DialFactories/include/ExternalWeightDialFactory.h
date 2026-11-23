/// @file ExternalWeightDialFactory.h
/// @brief Declares the factory and workers of `ExternalWeight` dials, whose
///        weights are computed by an external process.
/// @ingroup DialDictionary

#ifndef GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H
#define GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H

#include "DialFactoryBase.h"

#include "ConfigUtils.h"
#include "BinSet.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class Event;

/// Computes the weights of an ExternalWeight collection outside the dials.
///
/// The worker keeps one weight per registered event, or one per bin with
/// `useBinnedWeights`, in a list shared with every ExternalWeightDispatcher
/// of the collection. Its inputs are event variables (`inputEventVarList`,
/// plus the binning variables in binned mode, where each input is the bin
/// centre) and the fit parameters of the collection's input buffer. Both are
/// passed through POSIX shared memory, so that an external process can read
/// them and write the weights back. A derived class implements
/// evaluateImpl(), which asks that process to recompute the weights.
///
/// @par Life cycle
/// 1. configure() reads the common options; initialize() runs the derived
///    class's checks.
/// 2. Loading: registerEvent() once per event (unbinned mode only).
/// 3. First updateWeights(): the event inputs are moved into shared memory,
///    then the weights are evaluated. Later calls evaluate only when a
///    parameter changed.
///
/// See docs/configuration/ExternalWeightDials.md for the configuration and
/// the worker protocol.
/// @ingroup DialDictionary
class ExternalWeightWorker : public JsonBaseClass {
public:
  /// Destroys the worker and its shared-memory buffers.
  ~ExternalWeightWorker() override = default;

  /// Adds one event and records its input variables. Called, through
  /// ExternalWeightDialFactory::makeDial(), from the data-loading threads.
  /// @param event_ The event; it must have every input variable.
  /// @return The position of the event's weight in getWeightList().
  /// @warning Throws in binned mode, or once the inputs were moved to shared
  ///          memory (first updateWeights()).
  std::size_t registerEvent(const Event& event_);

  /// Recomputes the weights for the current parameter values, if needed.
  ///
  /// The first call finishes the loading (inputs copied to shared memory)
  /// and always evaluates. Later calls evaluate only when the buffer reports
  /// a changed input (DialInputBuffer::isDialUpdateRequested()). Called by
  /// the update callback of the collection (DialCollection::update()).
  /// @param inputBuffer_ The collection's input buffer, with the parameters
  ///        listed in its `dialInputList`.
  void updateWeights(DialInputBuffer& inputBuffer_);

  /// Creates the dial of one bin, in binned mode.
  /// @param binIndex_ Bin index in getBinning().
  /// @return An ExternalWeightDispatcher reading that bin's weight.
  /// @warning Throws if binned mode is off or the index is out of range.
  [[nodiscard]] virtual std::unique_ptr<DialBase> makeBinnedDial(std::size_t binIndex_);

  /// Tells whether there is one weight per bin instead of one per event.
  /// @return The `useBinnedWeights` option (default false).
  [[nodiscard]] bool useBinnedWeights() const { return _useBinnedWeights_; }
  /// Gives the binning of binned mode.
  /// @return The `binning` option; empty in unbinned mode.
  [[nodiscard]] const BinSet& getBinning() const { return _binning_; }

  /// Gives the event variables the worker reads.
  /// @return `inputEventVarList` without brackets, then, in binned mode, the
  ///         binning variables not already listed.
  [[nodiscard]] const std::vector<std::string>& getInputEventVarNameList() const { return _inputEventVarNameList_; }
  /// Gives the weight list shared with the dispatchers.
  /// @return The list; its entries are 1 until the first evaluation.
  [[nodiscard]] const std::shared_ptr<std::vector<double>>& getWeightList() const { return _weightList_; }

protected:
  /// A block of doubles in POSIX shared memory, created, mapped and removed
  /// with the object. The name is visible to other processes as `/NAME`.
  struct SharedMemoryBuffer {
    /// Creates and maps the block.
    /// @param name_ Name without the leading slash; it must not exist yet.
    /// @param nbDoubles_ Number of doubles; at least one is allocated.
    /// @warning Throws if the block can't be created, resized or mapped.
    explicit SharedMemoryBuffer(std::string name_, std::size_t nbDoubles_);
    /// Unmaps, closes and unlinks the block.
    ~SharedMemoryBuffer();

    /// Not copyable: the object owns the mapping.
    SharedMemoryBuffer(const SharedMemoryBuffer&) = delete;
    /// Not copyable: the object owns the mapping.
    /// @return Never returns.
    SharedMemoryBuffer& operator=(const SharedMemoryBuffer&) = delete;

    std::string name{}; ///< Name without the leading slash.
    std::size_t nbDoubles{0}; ///< Number of doubles requested.
    std::size_t nbBytes{0}; ///< Size of the mapping in bytes (at least one double).
    int fd{-1}; ///< File descriptor from shm_open(); -1 if not open.
    double* ptr{nullptr}; ///< Start of the mapped block.
  };

  /// Gives the names of the event inputs, in shared-memory buffer order.
  /// @return Same list as getInputEventVarNameList().
  [[nodiscard]] const std::vector<std::string>& getLoadedInputNameList() const { return _inputEventVarNameList_; }
  /// Gives the shared-memory buffers of the event inputs.
  /// @return One buffer per input name; empty before the first updateWeights().
  [[nodiscard]] const std::vector<std::unique_ptr<SharedMemoryBuffer>>& getInputBufferList() const { return _inputBufferList_; }
  /// Gives the shared-memory buffer of the parameter values.
  /// @return The buffer; null before the first evaluation.
  [[nodiscard]] const SharedMemoryBuffer* getParameterBuffer() const { return _parameterBuffer_.get(); }
  /// Gives the shared-memory buffer the external process writes the
  /// weights to.
  /// @return The buffer; null before the first updateWeights().
  [[nodiscard]] const SharedMemoryBuffer* getWeightBuffer() const { return _weightBuffer_.get(); }
  /// Gives the number of registered events.
  /// @return The event count, set when loading is finished; 0 in binned mode.
  [[nodiscard]] std::size_t getEventCount() const { return _eventCount_; }
  /// Gives the number of weights.
  /// @return The number of events, or of bins in binned mode.
  [[nodiscard]] std::size_t getWeightCount() const { return _weightCount_; }

protected:
  /// Removes spaces and one pair of surrounding brackets from an input name
  /// (`[Enu]` becomes `Enu`).
  /// @param inputName_ The name as written in the configuration.
  /// @return The bare variable name.
  static std::string normalizeInputName(const std::string& inputName_);
  /// Reads `inputEventVarList`, `useBinnedWeights` and `binning`. In binned
  /// mode, the weights start at 1, one per bin.
  /// @warning Throws for an empty input name, or in binned mode without a
  ///          binning or with an empty one.
  void configureImpl() override;

private:
  /// Asks the external code to fill the weight buffer from the input and
  /// parameter buffers.
  /// @param inputBuffer_ The collection's input buffer.
  virtual void evaluateImpl(const DialInputBuffer& inputBuffer_) = 0;
  /// Copies the parameter values to shared memory, runs evaluateImpl(), and
  /// copies the weights to getWeightList().
  /// @param inputBuffer_ The collection's input buffer.
  /// @pre initialize() was called and loading is finished.
  void evaluate(const DialInputBuffer& inputBuffer_);
  /// Moves the event inputs (or, in binned mode, the bin centres) into shared
  /// memory and creates the weight buffer, filled with 1. Runs once.
  void finalizeEventLoading();

  /// Bare names of the event inputs.
  std::vector<std::string> _inputEventVarNameList_{};
  /// Values of each input for each registered event, before they are moved
  /// to shared memory.
  std::vector<std::vector<double>> _inputEventValueList_{};
  /// Weights read by the dispatchers, one per event or bin.
  std::shared_ptr<std::vector<double>> _weightList_{std::make_shared<std::vector<double>>()};
  /// Shared-memory copy of each input, one value per event or bin.
  std::vector<std::unique_ptr<SharedMemoryBuffer>> _inputBufferList_{};
  /// Shared-memory copy of the parameter values.
  std::unique_ptr<SharedMemoryBuffer> _parameterBuffer_{nullptr};
  /// Shared memory where the external process writes the weights.
  std::unique_ptr<SharedMemoryBuffer> _weightBuffer_{nullptr};
  /// Number of registered events; set when loading is finished.
  std::size_t _eventCount_{0};
  /// Number of weights (events, or bins in binned mode).
  std::size_t _weightCount_{0};
  /// `useBinnedWeights` option.
  bool _useBinnedWeights_{false};
  /// `binning` option, for binned mode.
  BinSet _binning_{};
  /// True once the inputs are in shared memory; registerEvent() is then
  /// refused.
  bool _areEventsLoaded_{false};
  /// Serialises registerEvent() between loading threads.
  mutable std::mutex _eventRegistrationMutex_{};
};

/// ExternalWeightWorker that runs a Python script in a child process.
///
/// The child runs a small bootstrap (embedded in the .cpp) with the
/// configured Python executable. The bootstrap loads `evalScript`, which
/// must define `run(command)` (or the older `run_worker()`) and may define
/// `configure(args)`; it attaches the shared-memory buffers as NumPy arrays
/// and forwards the commands `initialize`, `evaluate` and `shutdown` to
/// `run`. Commands and replies are single JSON lines on the child's stdin
/// and stdout; every reply must have `"status": "ok"`.
///
/// Options, in `workerConfig`: `pythonExecutable` and `evalScript`
/// (required), `scriptArgs` (passed to the script), `initScript` (read but
/// ignored, with a warning).
/// @ingroup DialDictionary
class ExternalWeightPythonWorker : public ExternalWeightWorker {
public:
  /// Stops the child process, if it was started.
  ~ExternalWeightPythonWorker() override;

protected:
  /// Reads the common options, then `workerConfig`.
  /// @warning The job stops if `pythonExecutable` or `evalScript` is
  ///          missing.
  void configureImpl() override;
  /// Checks that the executable can be run and the script read, and
  /// compiles the script with `python -m py_compile`.
  /// @warning Throws if a check fails.
  void initializeImpl() override;

private:
  /// Starts the child process and sends it the `initialize` command, which
  /// describes the shared-memory buffers and the parameters.
  /// @param inputBuffer_ The collection's input buffer, for the parameter
  ///        names.
  void startWorkerProcess(const DialInputBuffer& inputBuffer_);
  /// Runs `python -m py_compile` on `evalScript` in a child process.
  /// @warning Throws with the compiler output if it fails.
  void validateEvalScript();
  /// Writes one JSON line to the child's stdin.
  /// @param command_ The command.
  void sendWorkerCommand(const JsonType& command_);
  /// Reads one JSON line from the child's stdout.
  /// @return The parsed reply.
  /// @warning Throws if the pipe closes or the line is not valid JSON.
  JsonType readWorkerResponse();
  /// Sends `shutdown` (ignoring any error), closes the pipes and waits for
  /// the child to exit.
  void stopWorkerProcess();
  /// Starts the child on first use, then sends `evaluate` and waits for the
  /// reply.
  /// @param inputBuffer_ The collection's input buffer.
  void evaluateImpl(const DialInputBuffer& inputBuffer_) override;
  std::string _pythonExecutable_{}; ///< `pythonExecutable`: interpreter to run.
  std::string _initScript_{}; ///< `initScript`: read but not used.
  std::string _evalScript_{}; ///< `evalScript`: user script defining `run(command)`.
  std::vector<std::string> _scriptArgs_{}; ///< `scriptArgs`: arguments passed to the script.
  bool _isWorkerStarted_{false}; ///< True while the child process runs.
  int _workerInputFd_{-1}; ///< Write end of the child's stdin pipe; -1 if closed.
  int _workerOutputFd_{-1}; ///< Read end of the child's stdout pipe; -1 if closed.
  int _workerPid_{-1}; ///< Process id of the child; -1 if none.
};

/// Dial factory of `dialType: ExternalWeight` collections.
///
/// DialCollection::initializeDialsWithExternalWeight() creates it from the
/// collection's `options` block (`type`, `inputEventVarList`,
/// `useBinnedWeights`, `binning`, `workerConfig`) and stores it as collection
/// data. The only worker `type` is `PythonWorker` (ExternalWeightPythonWorker).
/// In unbinned mode, DataDispenser calls makeDial() for each event; in binned
/// mode, the collection creates one dial per bin with makeBinnedDial(). The
/// collection's update callback calls updateWeights().
/// @ingroup DialDictionary
class ExternalWeightDialFactory : public DialFactoryBase, public JsonBaseClass {

public:
  /// Creates a factory without a worker; configure() creates it.
  ExternalWeightDialFactory() = default;

protected:
  /// Checks the options and creates and configures the worker.
  /// @warning Throws for an unknown worker `type`.
  void configureImpl() override;
  /// Initializes the worker. Called by DialCollection::initializeImpl().
  void initializeImpl() override;

public:

  /// Registers an event with the worker and creates its dial.
  /// @param event_ The event being loaded.
  /// @return A new ExternalWeightDispatcher for the event's weight; the
  ///         caller takes ownership.
  [[nodiscard]] DialBase* makeDial(const Event& event_) override;
  /// Creates the dial of one bin (ExternalWeightWorker::makeBinnedDial()).
  /// @param binIndex_ Bin index in getBinning().
  /// @return The dial.
  [[nodiscard]] std::unique_ptr<DialBase> makeBinnedDial(std::size_t binIndex_){ return _worker_->makeBinnedDial(binIndex_); }
  /// Recomputes the weights if needed (ExternalWeightWorker::updateWeights()).
  /// @param inputBuffer_ The collection's first input buffer.
  void updateWeights(DialInputBuffer& inputBuffer_){ _worker_->updateWeights(inputBuffer_); }

  /// Tells whether the weights are binned.
  /// @return ExternalWeightWorker::useBinnedWeights().
  /// @warning Throws if configure() has not created the worker.
  [[nodiscard]] bool useBinnedWeights() const{
    LogThrowIf(_worker_ == nullptr, "Can't fetch binned-weight mode. Worker isn't set.");
    return _worker_->useBinnedWeights();
  }

  /// Gives the binning of binned mode.
  /// @return ExternalWeightWorker::getBinning().
  /// @warning Throws if configure() has not created the worker.
  [[nodiscard]] const BinSet& getBinning() const{
    LogThrowIf(_worker_ == nullptr, "Can't fetch binning. Worker isn't set.");
    return _worker_->getBinning();
  }

  /// Gives the event variables the worker needs; the collection registers
  /// them as extra leaves.
  /// @return ExternalWeightWorker::getInputEventVarNameList().
  /// @warning Throws if configure() has not created the worker.
  [[nodiscard]] const std::vector<std::string>& getInputEventVarNameList() const{
    LogThrowIf(_worker_ == nullptr, "Can't fetch input event name list. Worker isn't set.");
    return _worker_->getInputEventVarNameList();
  }

private:
  /// The worker that computes the weights; it owns the event inputs and the
  /// weights.
  std::unique_ptr<ExternalWeightWorker> _worker_{nullptr};

};

#endif // GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H
