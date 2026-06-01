# Numeric exactness plan

The present schema layer stores number bounds and `multipleOf` as `f64`.  This is
not mathematically ideal because JSON Schema numbers are decimal JSON literals,
not IEEE-754 floats.

## Target type

Introduce:

```rust
struct DecimalRational {
    sign: Sign,
    numerator: BigInt,
    denominator: BigInt,
}
```

or a smaller decimal-specific representation:

```rust
struct Decimal {
    coefficient: BigInt,
    scale: u32,
}
```

Then represent:

```text
minimum, maximum: Decimal
exclusive flags: bool
multiple_of: PositiveDecimal
```

## Lowering exactness

For integer schemas, convert bounds to integer intervals exactly.  For number
schemas, construct JSON numeric regexes that match decimal values satisfying the
rational constraints or reject shapes where exact regex construction is not yet
available.

## Publication fallback

If exact decimal arithmetic is deferred, document the current f64 behavior as a
known precision limitation and add tests around decimal values that are commonly
misrepresented.
