## How to write JSON

All of the configuration files used in the Super-xsllh framework use JSON to specify options and values. This was chosen because JSON is simple and fairly readable. JSON stands for JavaScript Object Notation. JSON files generally have the `.json` extension.

JSON only has a few syntax rules:

+ Data are in key/value pairs
+ Key/value pairs are separated by a colon
+ Keys must be strings
+ Data are separated by commas
+ Curly braces hold objects
+ Square brackets hold arrays

JSON natively supports the following basic data types:

+ Numbers: always read by the parser as a floating point number, can use exponential E notation (e.g. `1.23E4`).
+ String: strings must be delimited with double-quotation marks (e.g. `"string"`).
+ Boolean: either of the values `true` or `false`.
+ Array: an ordered list of zero or more values, each of which may be any type. Arrays use square bracket notation and elements are comma-separated.
+ Object: an unordered collection of key-value pairs, where the names must be strings. Objects use curly braces. Similar to a Python dictionary or C++ map.
+ Null: an empty value, using the word `null`.

An example of a very basic JSON file:
```json
{
    "some_number" : 123,
    "some_string" : "example",
    "a_boolean" : true,
    "array_of_numbers" : [1234, 531, 641, 788],
    "an_object" : { "a key" : false, "second_key" : 4124, "array_key" : ["nested", "array"] }
}
```
Arrays and Objects can be nested within each other. The final element of an object or an array does not have a trailing comma. The intial opening and closing curly brace is to start a JSON object to hold the data, and are required. White space in general does not matter, except inside strings.

A more complicated example:
```json
{
    "firstName": "John",
    "lastName": "Smith",
    "isAlive": true,
    "age": 27,
    "address": {
        "streetAddress": "21 2nd Street",
        "city": "New York",
        "state": "NY",
        "postalCode": "10021-3100"
    },
    "phoneNumbers": [
        {
            "type": "home",
            "number": "212 555-1234"
        },
        {
            "type": "office",
            "number": "646 555-4567"
        },
        {
            "type": "mobile",
            "number": "123 456-7890"
        }
    ],
    "children": [],
    "spouse": null
}
```

### Parsing

The Super-xsllh framework uses [JSON for Modern C++](https://github.com/nlohmann/json) for parsing JSON files, which is included as a header-only library. Visit the GitHub page for the documentation on how to use it.

### References

+ [https://www.w3schools.com/whatis/whatis_json.asp](https://www.w3schools.com/whatis/whatis_json.asp)
+ [https://en.wikipedia.org/wiki/JSON](https://en.wikipedia.org/wiki/JSON)
