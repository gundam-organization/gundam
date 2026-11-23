/// @file Kriged.h
/// @brief Declares Kriged, an event-by-event dial that sums weighted entries
///        of a shared table.
/// @ingroup DialDictionary

#ifndef GUNDAM_KRIGED_H
#define GUNDAM_KRIGED_H

#include "DialBase.h"

/// A dial reweight an event based on an indexed table.  It is intended to
/// apply linear Kriging, but can be used for any linear reweighting that can
/// be represented as indices into a table, plus weights to apply to the table
/// value at the index.  The table is filled before the propagation by
/// KrigedDialFactory::update().  Notice that the input parameters are ignored
/// here since they were used to fill the table (See
/// KrigedDialFactory::update()).
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). KrigedDialFactory::makeDial() creates one Kriged dial per event
/// while DataDispenser loads the data: the factory's user library computes,
/// from the event's variables, which table entries the event depends on and
/// with which weights. The response is then `sum_i weight_i * table[index_i]`.
///
/// The table is shared by all dials of the collection and is owned by the
/// factory. It is refilled for each new set of parameter values by
/// KrigedDialFactory::updateTable(), which DialCollection::update() calls
/// (from DialManager::updateDialState()) through the callback registered in
/// DialCollection::initializeDialsWithKriging().
///
/// The GPU counterpart is Cache::Weight::Kriged.
/// @note The text above refers to `KrigedDialFactory::update()`; the function
///       that fills the table is KrigedDialFactory::updateTable().
/// @ingroup DialDictionary
class Kriged : public DialBase {

public:
    /// Destroys the dial.
    /// The table is not owned and is left untouched.
    ~Kriged() = default;

    /// Deleted: a Kriged dial always needs a table and its weights.
    Kriged() = delete;

    /// Creates a dial from the first `entries_` (index, weight) pairs.
    /// @param table_ Table owned by the KrigedDialFactory; it must outlive
    ///        the dial.
    /// @param entries_ Number of pairs to keep; `index_` and `weight_` must
    ///        hold at least this many values.
    /// @param index_ Table indices. Only the first `entries_` are read.
    /// @param weight_ Weights, matched with `index_`. They are stored as float.
    /// @param options_ Not used.
    /// @note `index_` and `weight_` are taken by value. KrigedDialFactory
    ///       passes its whole work buffers (at least 10000 entries each), so
    ///       both buffers are copied for every dial created.
    Kriged(const std::vector<double>* table_,
           int entries_, std::vector<int> index_, std::vector<double> weight_,
           const std::string& options_="");

    /// Copies this dial.
    /// @return A new Kriged with a copy of the (index, weight) pairs and the
    ///         same, shared, table pointer.
    [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Kriged>(*this); }

    /// Gives the type name of this dial.
    /// @return `"Kriged"`. Cache::Manager::Build() matches this name to size
    ///         the kriged part of the GPU cache.
    [[nodiscard]] std::string getDialTypeName() const override { return {"Kriged"}; }

    /// Calculate the response for this dial.  The DialInputBuffer isn't
    /// used for Kriging since it is reweighting based on a table that is
    /// filled before the parameters are propagated.
    ///
    /// @param input_ Not used. The parameters act only through the table
    ///        contents. The buffer still matters elsewhere: the dial is only
    ///        re-evaluated when its values change.
    /// @return The sum of `weight * table[index]` over the stored pairs.
    ///         The indices are not bounds-checked.
    [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

    /// Provide the internal information for the dial.  These could (should) be
    /// private and add the Cache::Manager classes as friends.
    ///
    /// @return Non-owning pointer to the shared table. Cache::Manager uses it
    ///         to find the table to copy to the GPU.
    const std::vector<double>* getTable() const { return _table_; }

    /// Gives the (table index, weight) pairs of this dial.
    /// @return The pairs. Cache::Manager::Build() counts them and
    ///         Cache::Manager::Update() copies them to the GPU.
    const std::vector<std::pair<int, float>>& getWeights ()const { return _weights_; }

private:
    /// A pointer to the table being used.  The table is owned by the
    /// associated KrigedDialFactory.
    const std::vector<double>* _table_{nullptr};

    /// Vector of the table indices and weights to apply to the table entry.
    /// The weight is a float that trades a tiny amount of numeric precision
    /// for a more efficient memory access.  A float saves 4 bytes and keeps
    /// the 32 bit alignment of the elements (which saves another 4 bytes).
    std::vector<std::pair<int,float>> _weights_;

};
#endif //GUNDAM_KRIGED_H
