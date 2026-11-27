# parse_duration

`parse_duration(text: &str) -> Result<u64, DurationError>` reads a short
duration and returns it in seconds.

## Grammar

- The text is one or more groups. A group is a number followed by a unit.
- A number is one or more ASCII digits (`0` to `9`). Leading zeros are
  allowed. There is no sign, no decimal point and no digit separator.
- A unit is one lower-case letter: `d` (86,400 seconds), `h` (3,600 seconds),
  `m` (60 seconds) or `s` (1 second).
- Units must appear in the order `d`, `h`, `m`, `s`. Each unit can appear at
  most once. Any unit can be left out.
- No other character is allowed. This includes spaces, upper-case letters
  and signs.

## Result

- The result is the sum of every number times the seconds of its unit.
- `0s` and `0h0m` are valid and give 0.

## Errors

| Input | Result |
| --- | --- |
| The empty string | `Err(DurationError::Empty)` |
| Anything that does not follow the grammar | `Err(DurationError::Invalid)` |
| A valid text whose number or sum does not fit in a `u64` | `Err(DurationError::Overflow)` |

## Examples

| Text | Result |
| --- | --- |
| `45s` | `Ok(45)` |
| `1h30m` | `Ok(5400)` |
| `2d3s` | `Ok(172803)` |
| `90m` | `Ok(5400)` |
| `1m1h` | `Err(DurationError::Invalid)` (wrong order) |
| `5` | `Err(DurationError::Invalid)` (no unit) |
| ` 5s` | `Err(DurationError::Invalid)` (space) |
