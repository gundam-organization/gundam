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
class Kriged : public DialBase {

public:
    ~Kriged() = default;
    Kriged() = delete;
    Kriged(const std::vector<double>* table_,
           int entries_, std::vector<int> index_, std::vector<double> weight_,
           const std::string& options_="");

    [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Kriged>(*this); }
    [[nodiscard]] std::string getDialTypeName() const override { return {"Kriged"}; }

    /// Calculate the response for this dial.  The DialInputBuffer isn't
    /// used for Kriging since it is reweighting based on a table that is
    /// filled before the parameters are propagated.
    [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

    /// Provide the internal information for the dial.  These could (should) be
    /// private and add the Cache::Manager classes as friends.
    const std::vector<double>* getTable() const { return _table_; }
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
