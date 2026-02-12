# Google Shell Style Guide for Claude Code

**Purpose**: Definitive style guide for Bash scripts, adapted for this project to ensure consistency and maintainability.

**Target Environment**: Bash (executables must start with `#!/bin/bash`).

**Philosophy**: Consistency, readability, and correctness.

**Source**: Adapted from [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

---

## Table of Contents

1. [Critical Safety & Design Patterns](#1-critical-safety--design-patterns)
2. [Formatting & Syntax](#2-formatting--syntax)
3. [Naming Conventions](#3-naming-conventions)
4. [Best Practices](#4-best-practices)
5. [Tooling](#5-tooling)

---

## 1. Critical Safety & Design Patterns

### 1.1 Executables vs Libraries

- **Executables**: Should have no extension (preferred) or `.sh`.
- **Libraries**: Must have a `.sh` extension and should not be executable.

### 1.2 SUID/SGID

**Forbidden**. SUID and SGID are forbidden on shell scripts due to security risks. Use `sudo` if elevated access is needed.

### 1.3 Eval

**Avoid**. `eval` munges input and makes state tracking difficult. Avoid it completely.

### 1.4 Pipes to While

**Use Process Substitution**. Pipes create subshells, so variables modified inside a piped loop won't persist.

**❌ BROKEN**:

```bash
last_line='NULL'
your_command | while read -r line; do
    last_line="${line}"
done
echo "${last_line}" # Output: NULL
```

**✅ CORRECT**:

```bash
last_line='NULL'
while read -r line; do
    last_line="${line}"
done < <(your_command)
echo "${last_line}" # Output: Last line content
```

---

## 2. Formatting & Syntax

### 2.1 Indentation

**2 Spaces**. No tabs.

### 2.2 Line Length

**80 Characters**. Split long lines using `\` or use here-docs for long strings.

### 2.3 Pipelines

Split one per line if they don't fit on a single line.

```bash
command1 \
  | command2 \
  | command3 \
  | command4
```

### 2.4 Control Flow

Put `; do` and `; then` on the same line as the `for`, `while`, or `if`.

**✅ CORRECT**:

```bash
if [[ "$var" == "value" ]]; then
    echo "match"
fi

for dir in "${dirs[@]}"; do
    echo "$dir"
done
```

### 2.5 Quoting

**Strongly Recommended Rules**:

- **Always quote** strings containing variables, command substitutions, spaces, or shell meta-characters.
- **Prefer** quoting strings that are "words".
- **Use Arrays** for lists of elements (especially flags).

```bash
# Variables
echo "${flag}"

# Command Substitution
flag="$(some_command "$@")"

# Literal Strings (no substitution)
echo 'Hello world'
```

### 2.6 Variable Expansion

Prefer `"${var}"` over `"$var"`.

**✅ CORRECT**:

```bash
echo "PATH=${PATH}, PWD=${PWD}, mine=${some_var}"
```

### 2.7 Command Substitution

Use `$(command)` instead of backticks `` `command` ``. Backticks are hard to nest and read.

**✅ CORRECT**:

```bash
var="$(command "$(command1)")"
```

### 2.8 Tests

Use `[[ ... ]]` over `[ ... ]` or `test`.
`[[ ... ]]` handles word splitting and path expansion safer, and supports regex.

**✅ CORRECT**:

```bash
if [[ -z "${my_var}" ]]; then
    do_something
fi
```

### 2.9 Arithmetic

Use `(( ... ))` or `$(( ... ))`. Never use `let` or `expr`.

**✅ CORRECT**:

```bash
(( i += 1 ))
val=$(( i * 5 ))
```

### 2.10 Arrays

Use Bash arrays for lists of items to avoid quoting issues.

**✅ CORRECT**:

```bash
declare -a flags
flags=(--foo --bar='baz')
mybinary "${flags[@]}"
```

---

## 3. Naming Conventions

### 3.1 Functions & Variables

**Snake Case**. Lowercase, with underscores to separate words.

- `my_func`
- `my_var`

### 3.2 Constants & Environment Variables

**Upper Case**. Separated with underscores. declared at the top of the file.

- `MY_CONSTANT`
- `PATH_TO_FILES`

### 3.3 Source Filenames

**Snake Case**. Lowercase, with underscores (e.g., `make_template.sh`).

### 3.4 Local Variables

**Use `local`**. Declare function-specific variables with `local` to avoid global namespace pollution. Ensure declaration and assignment are separate if using command substitution.

```bash
my_func() {
  local result
  result="$(command)"
}
```

---

## 4. Best Practices

### 4.1 Environment

**STDOUT vs STDERR**: All error messages should go to STDERR.

```bash
err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}
```

### 4.2 Comments

- **File Header**: Every file must have a top-level comment describing its contents.
- **Function Comments**: Any non-trivial function must have a header comment describing globals, arguments, outputs, and returns.
- **TODOs**: Use `TODO(user): description` format.

### 4.3 Return Values

**Always Check**. Check return values of commands, especially for file I/O and pipeline operations.

```bash
mv "${src}" "${dest}"
if (( $? != 0 )); then
  echo "Move failed" >&2
  exit 1
fi
```

### 4.4 Builtin Commands

**Prefer Builtins**. Use bash builtins (e.g., parameter expansion) over external commands (e.g., `sed`, `awk`) when possible for performance.

### 4.5 Main Function

**Required**. For scripts with at least one other function, put the main logic in a `main` function at the bottom.

```bash
main() {
  # ... code ...
}

main "$@"
```

---

## 5. Tooling

### 5.1 ShellCheck

**Recommended**. Use [ShellCheck](https://www.shellcheck.net/) to identify common bugs and warnings.
