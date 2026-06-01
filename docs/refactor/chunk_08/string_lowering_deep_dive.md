# String lowering deep dive
String lowerers work across three languages:

1. decoded Unicode scalar strings;
2. JSON string body byte sequences;
3. quoted JSON string texts.

JSON Schema `pattern` applies to the decoded string.  The grammar accepts encoded
JSON string texts.  The lowering from decoded regex HIR to JSON body regex is
therefore a semantic compiler, not a formatting helper.  It needs its own proof
comments and tests, especially for Unicode classes, anchors, dot, shorthand
classes, and escaped JSON characters.
