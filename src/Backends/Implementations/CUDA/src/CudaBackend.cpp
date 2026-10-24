#include "CudaBackend.h"

#ifdef GUNDAM_USING_CUDA
#include "CudaDeviceContext.h"
#endif

#include <limits>
#include <stdexcept>
#include <utility>

struct Backends::CudaBackendImpl {
  std::shared_ptr<const EngineView> engineView{};
#ifdef GUNDAM_USING_CUDA
  std::unique_ptr<CudaDeviceContext> context{};
#endif
};

Backends::CudaBackend::CudaBackend() : _impl_(std::make_unique<CudaBackendImpl>()) {}
Backends::CudaBackend::~CudaBackend() = default;

bool Backends::CudaBackend::isCompiled() {
#ifdef GUNDAM_USING_CUDA
  return true;
#else
  return false;
#endif
}

void Backends::CudaBackend::configureImpl() {
  // Reconfiguration invalidates the model and any device resources.
  _impl_ = std::make_unique<CudaBackendImpl>();
  _deviceIndex_ = 0;
  _config_.clearFields();
  _config_.defineFields({{"name"}, {"deviceIndex"}});
  _config_.checkConfiguration();
  // fillValue deliberately tolerates conversion errors. Reject a malformed
  // explicit selection instead of silently falling back to device zero.
  if( const auto* deviceIndex = _config_.getJsonEntry("deviceIndex") ){
    if( not deviceIndex->is_number_integer() or *deviceIndex < 0
        or *deviceIndex > std::numeric_limits<int>::max() ){
      throw std::runtime_error("CudaBackend: deviceIndex must be a nonnegative integer within the int range.");
    }
  }
  _config_.fillValue(_deviceIndex_, "deviceIndex");
}

void Backends::CudaBackend::initializeDevice() {
#ifdef GUNDAM_USING_CUDA
  if( not _impl_->context ){ _impl_->context = std::make_unique<CudaDeviceContext>(_deviceIndex_); }
#else
  throw std::runtime_error("CUDA support was not compiled. Configure GUNDAM with -DWITH_CUDA_LIB=ON and a CUDA toolkit.");
#endif
}

void Backends::CudaBackend::initializeImpl() { initializeDevice(); }

Backends::BackendCapabilities Backends::CudaBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsGpu = isCompiled();
  // No numerical outputs are implemented yet; do not advertise them.
  out.deviceName = isCompiled() ? "CUDA not initialized" : "CUDA support not compiled";
#ifdef GUNDAM_USING_CUDA
  if( _impl_->context ){ out.deviceName = _impl_->context->getDeviceName(); }
#endif
  return out;
}

const Backends::EngineView& Backends::CudaBackend::getEngineView() const {
  if( not _impl_->engineView ){ throw std::runtime_error("CudaBackend has not been built."); }
  return *_impl_->engineView;
}

void Backends::CudaBackend::build(const EngineView& engineView_) {
  buildShared(std::make_shared<EngineView>(engineView_));
}

void Backends::CudaBackend::buildShared(std::shared_ptr<const EngineView> engineView_) {
  _impl_->engineView.reset();
  if( not engineView_ ){ throw std::runtime_error("CudaBackend requires a model."); }
  // BackendManager calls buildShared directly after configure(). Direct users
  // may also use initialize() to check device availability before building.
  initializeDevice();
  _impl_->engineView = std::move(engineView_);
}

Backends::PropagationToken Backends::CudaBackend::requestPropagation(const PropagationInputs&) {
  if( not _impl_->engineView ){ throw std::runtime_error("CudaBackend has not been built."); }
  throwNoResult();
}

Backends::PropagationStatus Backends::CudaBackend::getStatus(const PropagationToken&) const {
  // No token can be issued until the propagation executor exists.
  return {};
}

bool Backends::CudaBackend::isReady(const PropagationToken&) const { return false; }

void Backends::CudaBackend::throwNoResult() {
  throw std::runtime_error("CudaBackend is a skeleton: CUDA propagation is not implemented; no result is available.");
}

void Backends::CudaBackend::wait(const PropagationToken&) { throwNoResult(); }
void Backends::CudaBackend::materialize(const PropagationToken&, OutputRequest) { throwNoResult(); }
double Backends::CudaBackend::getLikelihood(const PropagationToken&) const { throwNoResult(); }
const std::vector<double>& Backends::CudaBackend::getEventWeightsHostView(const PropagationToken&) const { throwNoResult(); }
const std::vector<double>& Backends::CudaBackend::getHistogramSumsHostView(const PropagationToken&) const { throwNoResult(); }
const std::vector<double>& Backends::CudaBackend::getHistogramSumSquaresHostView(const PropagationToken&) const { throwNoResult(); }
const Backends::EventState& Backends::CudaBackend::getEventStateHostView(const PropagationToken&) const { throwNoResult(); }
const std::vector<double>& Backends::CudaBackend::getOutsideSumsHostView(const PropagationToken&) const { throwNoResult(); }
const std::vector<double>& Backends::CudaBackend::getOutsideSumSquaresHostView(const PropagationToken&) const { throwNoResult(); }
Backends::BackendDeviceView Backends::CudaBackend::getDeviceView(const PropagationToken&) const { throwNoResult(); }
