# AGENTS.md
Single-header C++17 library. Build with cmake. See README.md for details.

## Build
`ash
cmake -B build
cmake --build build
`
"@

 = @"
# CLAUDE.md
Single-header C++17 library in the llm-* suite by Mattbusel.
- Header: include/{name}.hpp
- All implementation under #ifdef {NAME}_IMPLEMENTATION
- No external deps except libcurl (if networking required)
- MIT License
