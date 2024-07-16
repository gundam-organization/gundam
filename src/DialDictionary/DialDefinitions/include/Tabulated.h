#ifndef GUNDAM_TABULATED_H
#define GUNDAM_TABULATED_H

#include "DialBase.h"

// Implement a dial to look up a weight in a table and apply it.  This is a
// very low level dial that depends on the Tabulated DialCollection to have
// properly filled the table.  Notice that the input parameters are ignored
// here since they were used to fill the table (See
// TabulatedDialFactor::update()).
class Tabulated : public DialBase {

public:
    ~Tabulated() = default;
    Tabulated() = delete;
    Tabulated(const std::vector<double>* table_, int index_, double fraction_,
        const std::string& options_="")
        : _table_{table_}, _fraction_{(float) fraction_}, _index_{index_} {};

    [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Tabulated>(*this); }
    [[nodiscard]] std::string getDialTypeName() const override { return {"Tabulated"}; }

    [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override
        { return _fraction_*(*_table_)[_index_] + (1.0-_fraction_)*(*_table_)[_index_+1]; }

    // Provide the internal information for the dial.  These could (should) be
    // private and add the Cache::Manager classes as friends.
    const std::vector<double>* getTable() const { return _table_; }
    int getIndex() const { return _index_; }
    double getFraction() const { return _fraction_; }

private:
    // A pointer to the table being used.  The table is owned by the
    // associated TabulatedDialFactory.
    const std::vector<double>* _table_{nullptr};

    // Interpolation between successive point tables.  This can be a float
    // since it is a fraction (zero to one) of the small difference between
    // table values, and keeps the 32 bit alignment of the following _index_
    // without wasting memory.  BUT, keep it before index for alignment (in
    // case changed to double).
    float _fraction_{0.0};

    // The precalculated point in the table to use for interpolation.
    int _index_{0};
};
#endif //GUNDAM_TABULATED_H
