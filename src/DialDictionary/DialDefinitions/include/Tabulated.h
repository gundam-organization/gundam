/// @file Tabulated.h
/// @brief Declares Tabulated, an event-by-event dial that interpolates
///        between two entries of a shared table.
/// @ingroup DialDictionary

#ifndef GUNDAM_TABULATED_H
#define GUNDAM_TABULATED_H

#include "DialBase.h"

/// Implement a dial to look up a weight in a table and apply it.  This is a
/// very low level dial that depends on the Tabulated DialCollection to have
/// properly filled the table.  Notice that the input parameters are ignored
/// here since they were used to fill the table (See
/// TabulatedDialFactor::update()).
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). TabulatedDialFactory::makeDial() creates one Tabulated dial per
/// event while DataDispenser loads the data. The factory's user library maps
/// the event's variables to a fractional table position; its integer part
/// becomes the index of this dial and its fractional part the fraction.
///
/// The table is shared by all dials of the collection and is owned by the
/// factory. It is refilled for each new set of parameter values by
/// TabulatedDialFactory::updateTable(), which DialCollection::update() calls
/// (from DialManager::updateDialState()) through the callback registered in
/// DialCollection::initializeDialsWithTabulation().
///
/// The GPU counterpart is Cache::Weight::Tabulated, which uses the same
/// formula as evalResponse().
/// @note `TabulatedDialFactor::update()` named above does not exist; the
///       function that fills the table is TabulatedDialFactory::updateTable().
/// @ingroup DialDictionary
class Tabulated : public DialBase {

public:
    /// Destroys the dial.
    /// The table is not owned and is left untouched.
    ~Tabulated() = default;

    /// Deleted: a Tabulated dial always needs a table position.
    Tabulated() = delete;

    /// Creates a dial that interpolates between table entries `index_` and
    /// `index_ + 1`.
    /// @param table_ Table owned by the TabulatedDialFactory; it must outlive
    ///        the dial.
    /// @param index_ Index of the first of the two entries; `index_ + 1` must
    ///        also be a valid index.
    /// @param fraction_ Weight of entry `index_` in evalResponse() (entry
    ///        `index_ + 1` gets `1 - fraction_`). It is rounded to float
    ///        precision before being stored.
    /// @param options_ Text returned by getSummary(). Only kept when compiled
    ///        with TABULATED_DEBUG_DIAL, otherwise ignored.
    ///        TabulatedDialFactory::makeDial() does not pass it.
    Tabulated(const std::vector<double>* table_, int index_, double fraction_,
              const std::string& options_="")
        : _table_{table_}, _fraction_{(float) fraction_}, _index_{index_} {
#ifdef TABULATED_DEBUG_DIAL
        _options_ = options_;
#endif
    };

    /// Copies this dial.
    /// @return A new Tabulated with the same index and fraction and the same,
    ///         shared, table pointer.
    [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Tabulated>(*this); }

    /// Gives the type name of this dial.
    /// @return `"Tabulated"`. Cache::Manager::Build() matches this name to
    ///         reserve table space on the GPU.
    [[nodiscard]] std::string getDialTypeName() const override { return {"Tabulated"}; }

    /// Interpolates between two consecutive table entries.
    /// @param input_ Not used. The parameters act only through the table
    ///        contents. The buffer still matters elsewhere: the dial is only
    ///        re-evaluated when its values change.
    /// @return `fraction * table[index] + (1 - fraction) * table[index + 1]`.
    /// @warning TabulatedDialFactory::makeDial() sets the fraction to the
    ///          fractional part of the bin position returned by the library,
    ///          and the factory documentation calls the integer part "the index
    ///          of the value below the value to be interpolated". With this
    ///          formula a fraction of 0 gives `table[index + 1]` and a fraction
    ///          close to 1 gives `table[index]`, the reverse of the usual linear
    ///          interpolation. The GPU kernel (Cache::Weight::Tabulated) uses
    ///          the same formula, so CPU and GPU agree.
    [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override
        { return _fraction_*(*_table_)[_index_] + (1.0-_fraction_)*(*_table_)[_index_+1]; }

    /// Provide the internal information for the dial.  These could (should) be
    /// private and add the Cache::Manager classes as friends.
    ///
    /// @return Non-owning pointer to the shared table. Cache::Manager uses it
    ///         to find the table to copy to the GPU.
    const std::vector<double>* getTable() const { return _table_; }

    /// Gives the index of the first of the two table entries used.
    /// @return The index, used by Cache::Manager::Update().
    int getIndex() const { return _index_; }

    /// Gives the interpolation fraction.
    /// @return The weight of entry getIndex() in evalResponse() (float
    ///         precision, returned as double). Used by
    ///         Cache::Manager::Update().
    double getFraction() const { return _fraction_; }

#ifdef TABULATED_DEBUG_DIAL
    /// Gives the options text passed to the constructor. Only compiled with
    /// TABULATED_DEBUG_DIAL, which is not defined anywhere in the repository.
    /// @return The options text.
    [[nodiscard]] virtual std::string getSummary() const override {
        return _options_;
    };
#endif

private:
    /// A pointer to the table being used.  The table is owned by the
    /// associated TabulatedDialFactory.
    const std::vector<double>* _table_{nullptr};

    /// Interpolation between successive point tables.  This could be a float
    /// since it is a fraction (zero to one) of the small difference between
    /// table values, and loses almost all of it's bits during the
    /// interpolation.  A float would save 8 bytes and keep the 32 bit
    /// alignment of the following _index_ without wasting memory.  BOTTOM
    /// LINE: keep it before index for alignment (safe for both float and
    /// double).
    ///
    /// @note The constructor casts the value to float before storing it, so it
    ///       has float precision even though it is a double.
    double _fraction_{0.0};

    /// The precalculated point in the table to use for interpolation.
    int _index_{0};

#ifdef TABULATED_DEBUG_DIAL
    /// Options text returned by getSummary() (TABULATED_DEBUG_DIAL builds
    /// only).
    std::string _options_;
#endif
};
#endif //GUNDAM_TABULATED_H
