const name = "Ada";
const escaped = `a backtick: \` and slash: \\`;
const interpolation = `hello ${name}, answer ${1 + 2}`;
function tag(strings, ...values) { return strings[0] + values[0]; }
const tagged = tag`value ${name}`;
const regex = /[A-Z]+\/[0-9]{2}/giu;
const divided = 12 / 3;