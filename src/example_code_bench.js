// Simple JavaScript code for benchmarking
const x = 12345;
let y = "hello world";
let z = true;
function add(a, b) {
    return a + b;
}
const result = add(x, 100);
if (result > 10000) {
    console.log("big");
} else {
    console.log("small");
}
for (let i = 0; i < 10; i++) {
    x = x + 1;
}
const obj = {
    name: "test",
    value: 123,
    flag: true
};
const arr = [1, 2, 3, 4, 5];
