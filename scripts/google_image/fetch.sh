#!/bin/bash
echo
index=0
out_dir=${1%.*}
mkdir -p "${out_dir}"

while IFS= read -r line; do
	index=$((index + 1))
	filename=${out_dir}/`printf %08d.jpg "${index}"`
	curl "${line}" -o "${filename}"

	property=$(identify "${filename}")
	if [[ ! $property =~ "JPEG" ]]; then
		rm "${filename}"
	fi
done < $1
