// Stress Test 1: Deeply Nested Structures
// Tests parser stack depth with extreme nesting

// 50 levels of nested function declarations
function f1() {
function f2() {
function f3() {
function f4() {
function f5() {
function f6() {
function f7() {
function f8() {
function f9() {
function f10() {
function f11() {
function f12() {
function f13() {
function f14() {
function f15() {
function f16() {
function f17() {
function f18() {
function f19() {
function f20() {
function f21() {
function f22() {
function f23() {
function f24() {
function f25() {
function f26() {
function f27() {
function f28() {
function f29() {
function f30() {
function f31() {
function f32() {
function f33() {
function f34() {
function f35() {
function f36() {
function f37() {
function f38() {
function f39() {
function f40() {
function f41() {
function f42() {
function f43() {
function f44() {
function f45() {
function f46() {
function f47() {
function f48() {
function f49() {
function f50() {
  return 42;
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

// 30 levels of nested objects
const deepObj = {
  a: { b: { c: { d: { e: { f: { g: { h: { i: { j: {
  k: { l: { m: { n: { o: { p: { q: { r: { s: { t: {
  u: { v: { w: { x: { y: { z: { aa: { bb: { cc: { dd: {
    value: 123
  }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
};

// 30 levels of nested arrays
const deepArr = [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ 
  1, 2, 3 
]]]]]]]]]]]]]]]]]]]]]]]]]]]]]];

// 30 levels of nested conditionals
function deepIf(x) {
  if (x > 0) {
  if (x > 1) {
  if (x > 2) {
  if (x > 3) {
  if (x > 4) {
  if (x > 5) {
  if (x > 6) {
  if (x > 7) {
  if (x > 8) {
  if (x > 9) {
  if (x > 10) {
  if (x > 11) {
  if (x > 12) {
  if (x > 13) {
  if (x > 14) {
  if (x > 15) {
  if (x > 16) {
  if (x > 17) {
  if (x > 18) {
  if (x > 19) {
  if (x > 20) {
  if (x > 21) {
  if (x > 22) {
  if (x > 23) {
  if (x > 24) {
  if (x > 25) {
  if (x > 26) {
  if (x > 27) {
  if (x > 28) {
  if (x > 29) {
    return x * 2;
  }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  return 0;
}

// Mixed nesting: function + object + array + conditional
function mixedNest(data) {
  return {
    result: (function() {
      return [
        data.map(function(item) {
          return {
            processed: (function() {
              if (item > 0) {
                return [
                  { value: item * 2 },
                  { doubled: item + item },
                  { computed: (function() {
                    return item * item;
                  })() }
                ];
              }
              return null;
            })()
          };
        })
      ];
    })()
  };
}

// Deeply nested ternary expressions (20 levels)
const ternary = (x > 0 ? (x > 1 ? (x > 2 ? (x > 3 ? (x > 4 ? 
  (x > 5 ? (x > 6 ? (x > 7 ? (x > 8 ? (x > 9 ? 
  (x > 10 ? (x > 11 ? (x > 12 ? (x > 13 ? (x > 14 ? 
  (x > 15 ? (x > 16 ? (x > 17 ? (x > 18 ? (x > 19 ? 
    'deep' : 'a') : 'b') : 'c') : 'd') : 'e') 
  : 'f') : 'g') : 'h') : 'i') : 'j')
  : 'k') : 'l') : 'm') : 'n') : 'o')
  : 'p') : 'q') : 'r') : 's') : 't');

// Deeply nested function calls (25 levels)
function wrap(f) { return function(x) { return f(x); }; }
const deepCall = wrap(wrap(wrap(wrap(wrap(
  wrap(wrap(wrap(wrap(wrap(
  wrap(wrap(wrap(wrap(wrap(
  wrap(wrap(wrap(wrap(wrap(
  wrap(wrap(wrap(wrap(wrap(
    function(x) { return x + 1; }
  ))))))))))))))))))))))));

// Export for module use
if (typeof module !== 'undefined') {
  module.exports = { f1, deepObj, deepArr, deepIf, mixedNest, deepCall };
}
