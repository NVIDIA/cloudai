#!/bin/bash

VERBOSE=true

if [ $# -ne 2 ]; then
    echo "Usage: $0 <test_scenario_path> <expected_output_path>"
    echo "test_scenario_path: the path of the test scenario"
    echo "expected_output_path: the path of the directory containing the expected result of the test"
    echo "Example: $0 conf/v0.6/general/test_scenario/nccl_test/test_scenario.toml ci_tools/functional_tests/scenarios_expected_outputs/nccl_test/"
    exit 1
fi

files_diff() {
    local file1="$1"
    local file2="$2"
    # Ignore:
	# - Empty lines
	# - Commented lines (start with #)
    diff <(grep -v '^\s*#' <(grep -v '^[[:space:]]*$' "$file1")) <(grep -v '^\s*#' <(grep -v '^[[:space:]]*$' "$file2")) > /dev/null
}

# recursively compare directories using files_diff
dirs_diff() {
    local error=false
    local dir1="$1"
    local dir2="$2"

    local files1=$(find "$dir1" -type f)
    local files2=$(find "$dir2" -type f)

    if [ "$(echo "$files1" | wc -l)" -ne "$(echo "$files2" | wc -l)" ]; then
        >&2 echo "Directories have different count of files."
        error=true
    fi

    while IFS= read -r file1; do
        local file2=$(echo "$file1" | sed "s|^$dir1|$dir2|")
        if [ ! -f "$file2" ]; then
            >&2 echo "File $file2 does not exist in $dir2."
			error=true
        fi

        if ! files_diff "$file1" "$file2"; then
            >&2 echo "Files $file1 and $file2 have different contents."
            error=true
        fi
    done <<< "$files1"

    if $error; then
        return 1
    fi
}


scenario_path="$1"
expected_output_path="$2"

if [ ! -f "$scenario_path" ]; then
    >&2 echo "Error: Scenario $scenario is not valid, can't find path $scenario_path."
    exit 1
fi

[ ! -d "results" ] && mkdir results

last_result_before=$(ls results/ -la -X | tail -n 3 | head -n 1 | awk '{print $NF}')

python main.py \
    --mode dry-run\
    --system_config_path "ci_tools/functional_tests/system_config.toml" \
    --test_scenario_path $scenario_path

last_result=$(ls results/ -la -X | tail -n 3 | head -n 1 | awk '{print $NF}')

if [ "$last_result_before" == "$last_result" ]; then
    >&2 echo "No new result added after running cloudai dry run."
    exit 1
fi

last_result_path="results/$last_result"

dirs_diff "$expected_output_path" "$last_result_path"
is_diff=$?

if [ $is_diff -eq 1 ]; then
    >&2 echo "Result output is not as expected."
    exit 1
fi

$VERBOSE && echo "Test ran successfully"

exit 0
