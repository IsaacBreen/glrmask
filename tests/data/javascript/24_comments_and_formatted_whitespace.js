const first = 1;
/* standalone block comment */
/* inline block comment */ const second = 2;
/* separated block comment */
function spaced(
    value
) {
    return value; /* trailing block comment */
}
const result = spaced(first + second);