#!/bin/bash

# Check for required arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <num_files> <output_folder> <atom_min> <atom_max>"
    echo "Example: $0 5 pdb_files 1000 2000"
    exit 1
fi

# Read command-line arguments
NUM_FILES=$1
OUTPUT_FOLDER=$2
ATOM_MIN=$3
ATOM_MAX=$4

# Ensure the output directory exists
mkdir -p "$OUTPUT_FOLDER"

# Corrected JSON query
QUERY=$(cat <<EOF
{
  "query": {
    "type": "terminal",
    "service": "text",
    "parameters": {
      "attribute": "rcsb_entry_info.deposited_atom_count",
      "operator": "range",
      "value": {
        "from": $ATOM_MIN,
        "include_lower": true,
        "to": $ATOM_MAX,
        "include_upper": true
      }
    }
  },
  "return_type": "entry",
  "request_options": {
    "return_all_hits": true
  }
}
EOF
)

# Fetch list of PDB entries matching the criteria
PDB_LIST_JSON=$(curl -s -X POST "https://search.rcsb.org/rcsbsearch/v2/query" -H "Content-Type: application/json" -d "$QUERY")

# Extract PDB IDs using jq
PDB_LIST=$(echo "$PDB_LIST_JSON" | jq -r '.result_set[].identifier')

# Convert to array (this handles multi-line output from jq)
PDB_ARRAY=($PDB_LIST)

# Check if we got results
if [ ${#PDB_ARRAY[@]} -eq 0 ]; then
    echo "No PDB entries found in the specified atom count range."
    exit 1
fi

echo "Found ${#PDB_ARRAY[@]} matching PDB entries. Downloading $NUM_FILES files..."

SUCCESS_COUNT=0
ATTEMPTS=0
MAX_ATTEMPTS=$((NUM_FILES * 3)) # Avoid infinite loops

# Keep downloading until we get NUM_FILES valid files
while [ $SUCCESS_COUNT -lt $NUM_FILES ] && [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    RANDOM_PDB=${PDB_ARRAY[$RANDOM % ${#PDB_ARRAY[@]}]}

    # Try downloading .cif.gz first
    CIF_FILE="${OUTPUT_FOLDER}/${RANDOM_PDB}.cif.gz"
    PDB_FILE="${OUTPUT_FOLDER}/${RANDOM_PDB}.pdb"

    echo -ne "[$((SUCCESS_COUNT+1))/$NUM_FILES] Downloading ${RANDOM_PDB}.cif.gz...\r"
    wget -q -O "$CIF_FILE" "https://files.rcsb.org/download/${RANDOM_PDB}.cif.gz"

    # Check if .cif.gz download was successful and is not empty
    if [ $? -eq 0 ] && [ -s "$CIF_FILE" ]; then
        echo -e "\n[$((SUCCESS_COUNT+1))/$NUM_FILES] Successfully downloaded ${RANDOM_PDB}.cif.gz"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "\n${RANDOM_PDB}.cif.gz failed or is empty. Trying .pdb..."
        rm -f "$CIF_FILE"

        # Try downloading .pdb
        wget -q -O "$PDB_FILE" "https://files.rcsb.org/download/${RANDOM_PDB}.pdb"

        # Check if .pdb download was successful and is not empty
        if [ $? -eq 0 ] && [ -s "$PDB_FILE" ]; then
            echo -e "\n[$((SUCCESS_COUNT+1))/$NUM_FILES] Successfully downloaded ${RANDOM_PDB}.pdb"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo -e "\n${RANDOM_PDB}.pdb also failed or is empty. Skipping..."
            rm -f "$PDB_FILE"
        fi
    fi

    ATTEMPTS=$((ATTEMPTS + 1))
done

if [ $SUCCESS_COUNT -lt $NUM_FILES ]; then
    echo "Warning: Only downloaded $SUCCESS_COUNT out of $NUM_FILES requested files."
else
    echo "Successfully downloaded $NUM_FILES files into $OUTPUT_FOLDER."
fi
