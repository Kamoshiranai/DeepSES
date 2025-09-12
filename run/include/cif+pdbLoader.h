#ifndef COORDS_H
#define COORDS_H
#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <cctype>
#include <gemmi/cif.hpp> // gemmi library for CIF parsing
#include <gemmi/gz.hpp>
#include <gemmi/mmcif.hpp>
//#include <gemmi/mmread.hpp>
#include <gemmi/model.hpp>

#include <glm/glm.hpp>
struct atom {
  glm::vec3 coords;
  float vdw_radius;
  glm::vec3 element_color;
  float b_factor;
};

class Protein {
public:
    Protein();
    int Load(std::string filename, std::string filetype);
    // Compute bounding box for a subset of atoms
    std::tuple<float, float, float, float, float, float>
    ComputeBounds(int num_atoms);
    // Scale the b-factors to lie between 0 and 1 for plotting purposes;
    void ScaleBFactors();
    void ParsePDB(std::ifstream& pdbFile);
    void ParseCIF(const std::string& filename);
    float LargestVdwRadius();

    std::vector<std::string> atomNames;
    std::vector<atom> atoms;
    int numAtoms;
};

// Simple trim function that removes leading and trailing whitespace
std::string trim(const std::string &s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(static_cast<unsigned char>(*start))) {
        ++start;
    }
    auto end = s.end();
    do {
        --end;
    } while (std::distance(start, end) > 0 && std::isspace(static_cast<unsigned char>(*end)));
    return std::string(start, end + 1);
}

Protein::Protein() { this->numAtoms = 0; }

int Protein::Load(std::string filename, std::string filetype) {
    try {
        if (filetype == "pdb") {
            std::ifstream pdbFile(filename);
            if (!pdbFile.is_open()) {
                throw std::invalid_argument("Error: File does not exist " + filename);
            }
            ParsePDB(pdbFile);
            pdbFile.close();
        } else if (filetype == "cif" || filetype == "cif.gz") {
            ParseCIF(filename);
        } else {
            throw std::invalid_argument("Unsupported file type: " + filetype);
        }
        ScaleBFactors();
    } catch (std::exception &e) {
        std::cerr << e.what() << '\n';
        return 0;
    }
    return 1;
}

void Protein::ParsePDB(std::ifstream& pdbFile) {
    int lineNumber = 0;
    for (std::string line; std::getline(pdbFile, line);) {
        lineNumber++;

        // Process only ATOM records
        if (line.substr(0,6) == "ENDMDL") break; // stop after the first model
        //if (line.substr(0,6) == "TER   ") break; // stop after the first chain
        if (line.substr(0,6) != "ATOM  ") continue;

        // Check if the line is long enough for a valid ATOM record.
        // PDB ATOM lines should be at least 66 characters long.
        if (line.size() < 66) {
          std::ostringstream oss;
            oss << "Malformed line (too short) at line " << lineNumber << ": " << line;
            throw std::invalid_argument(oss.str());
        }

        try {
          // Extract fields using fixed column positions (columns are 1-indexed)
          int seqNum = std::stoi(line.substr(6,5));           // columns 7-11
          std::string rawAtomName = line.substr(12,4);             // columns 13-16
          std::string atomName = trim(rawAtomName); // remove leading and trailing whitespace
          char altLoc = (line.size() >= 17) ? line[16] : ' ';     // column 17
          std::string residueName = line.substr(17,3);          // columns 18-20
          std::string chain = line.substr(21,1);                // column 22
          int residueNum = std::stoi(line.substr(22,4));        // columns 23-26
          float coordx = std::stof(line.substr(30,8));          // columns 31-38
          float coordy = std::stof(line.substr(38,8));          // columns 39-46
          float coordz = std::stof(line.substr(46,8));          // columns 47-54
          float occupancy = std::stof(line.substr(54,6));       // columns 55-60
          float temperature = std::stof(line.substr(60,6));     // columns 61-66

          // Ignore alternate locations (keep only ' ' or 'A')
          if (altLoc != ' ' && altLoc != 'A') {
            continue;  // Skip this atom
          }

          atom new_atom;
          new_atom.coords = glm::vec3(coordx, coordy, coordz);
          new_atom.b_factor = temperature;
          this->atomNames.push_back(atomName);

          //   Save vdw radii and color based on atom type
          // Zinc
          if (atomName.substr(0,2) == "ZN") {
            new_atom.vdw_radius = 1.39;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Magnesium
          } else if (atomName.substr(0,2) == "MG") {
            new_atom.vdw_radius = 1.73;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Mangan
          } else if (atomName.substr(0,2) == "MN") {
            new_atom.vdw_radius = 1.97;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Copper
          } else if (atomName.substr(0,2) == "CU") {
            new_atom.vdw_radius = 1.4;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Silicon
          } else if (atomName.substr(0,2) == "SI") {
            new_atom.vdw_radius = 2.1;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Iron
          } else if (atomName.substr(0,2) == "FE") {
            new_atom.vdw_radius = 2.44; //NOTE: not sure if this is right
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Aluminium
          } else if (atomName.substr(0,2) == "AL") {
            new_atom.vdw_radius = 1.84; //NOTE: not sure if this is right
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Beryllium
          } else if (atomName.substr(0,2) == "BE") {
            new_atom.vdw_radius = 1.53; //NOTE: not sure if this is right
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Carbon
          } else if (atomName[0] == 'C') {
            new_atom.vdw_radius = 1.7;
            new_atom.element_color = glm::vec3(0.2f, 0.2f, 0.2f);
          // Hydrogen
          } else if (atomName[0] == 'H') {
            new_atom.vdw_radius = 1.2;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 1.0f);
            // Nitrogen
          } else if (atomName[0] == 'N') {
            new_atom.vdw_radius = 1.55;
            new_atom.element_color = glm::vec3(0.0f, 0.0f, 1.0f);
            // Oxygen
          } else if (atomName[0] == 'O') {
            new_atom.vdw_radius = 1.52;
            new_atom.element_color = glm::vec3(1.0f, 0.0f, 0.0f);
            // Phosphor
          } else if (atomName[0] == 'P') {
            new_atom.vdw_radius = 1.8;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Sulfur
          } else if (atomName[0] == 'S') {
            new_atom.vdw_radius = 1.8;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Flourine
          } else if (atomName[0] == 'F') {
            new_atom.vdw_radius = 1.47;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Potassium
          } else if (atomName[0] == 'K') {
            new_atom.vdw_radius = 2.75;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
            // Iodine
          } else if (atomName[0] == 'I') {
            new_atom.vdw_radius = 1.98;
            new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
          } else {
            new_atom.vdw_radius = 1.2;
            new_atom.element_color = glm::vec3(0.0f, 1.0f, 0.0f);
            std::cout
                << "WARNING: vdw radius of " << atomName
                << " not known. Use 1.2 as default. Default color is green"
                << std::endl;
          }
            
          this->atoms.push_back(new_atom);
          this->numAtoms++;
        } catch (const std::exception &e) {
          std::ostringstream oss;
          oss << "Error parsing line " << lineNumber << ": " << line 
              << "\nException: " << e.what();
          throw std::invalid_argument(oss.str());
        }
    }
}

void Protein::ParseCIF(const std::string& filename) {
    gemmi::cif::Document doc = gemmi::cif::read(gemmi::MaybeGzipped(filename));
    //gemmi::Structure structure = gemmi::read_structure(filename);
    //gemmi::Structure structure = gemmi::read_structure_file(filename);

    // gemmi::Structure structure = gemmi::make_structure(doc);
    gemmi::Structure structure = gemmi::make_structure_from_block(doc.sole_block());
    
    //for (const auto& model : structure.models) {
    // Only process the first model if multiple are present.
    if (!structure.models.empty()) {
        const auto& model = structure.models.front();
        // // Process only the first chain in the model
        // if (!model.chains.empty()) {
        //     const auto& chain = model.chains.front();
        for (const auto& chain : model.chains) {
            for (const auto& residue : chain.residues) {
                for (const auto& Atom : residue.atoms) {

                    // Skip atoms with alternate locations other than ' ' or 'A'
                    if (Atom.has_altloc() && Atom.altloc != 'A') {
                        continue;
                    }

                    atom new_atom;
                    new_atom.coords = glm::vec3(Atom.pos.x, Atom.pos.y, Atom.pos.z);
                    new_atom.b_factor = Atom.b_iso;
                    this->atomNames.push_back(Atom.name);
                    
                    //   Save vdw radii and color based on atom type
                    // Zinc
                    if (Atom.name.substr(0,2) == "ZN") {
                      new_atom.vdw_radius = 1.39;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Magnesium
                    } else if (Atom.name.substr(0,2) == "MG") {
                      new_atom.vdw_radius = 1.73;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Mangan
                    } else if (Atom.name.substr(0,2) == "MN") {
                      new_atom.vdw_radius = 1.97;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Copper
                    } else if (Atom.name.substr(0,2) == "CU") {
                      new_atom.vdw_radius = 1.4;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Silicon
                    } else if (Atom.name.substr(0,2) == "SI") {
                      new_atom.vdw_radius = 2.1;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Iron
                    } else if (Atom.name.substr(0,2) == "FE") {
                      new_atom.vdw_radius = 2.44; //NOTE: not sure if this is right
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Aluminium
                    } else if (Atom.name.substr(0,2) == "AL") {
                      new_atom.vdw_radius = 1.84; //NOTE: not sure if this is right
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Beryllium
                    } else if (Atom.name.substr(0,2) == "BE") {
                      new_atom.vdw_radius = 1.53; //NOTE: not sure if this is right
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Carbon
                    } else if (Atom.name[0] == 'C') {
                      new_atom.vdw_radius = 1.7;
                      new_atom.element_color = glm::vec3(0.2f, 0.2f, 0.2f);
                    // Hydrogen
                    } else if (Atom.name[0] == 'H') {
                      new_atom.vdw_radius = 1.2;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 1.0f);
                      // Nitrogen
                    } else if (Atom.name[0] == 'N') {
                      new_atom.vdw_radius = 1.55;
                      new_atom.element_color = glm::vec3(0.0f, 0.0f, 1.0f);
                      // Oxygen
                    } else if (Atom.name[0] == 'O') {
                      new_atom.vdw_radius = 1.52;
                      new_atom.element_color = glm::vec3(1.0f, 0.0f, 0.0f);
                      // Phosphor
                    } else if (Atom.name[0] == 'P') {
                      new_atom.vdw_radius = 1.8;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Sulfur
                    } else if (Atom.name[0] == 'S') {
                      new_atom.vdw_radius = 1.8;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Flourine
                    } else if (Atom.name[0] == 'F') {
                      new_atom.vdw_radius = 1.47;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Potassium
                    } else if (Atom.name[0] == 'K') {
                      new_atom.vdw_radius = 2.75;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                      // Iodine
                    } else if (Atom.name[0] == 'I') {
                      new_atom.vdw_radius = 1.98;
                      new_atom.element_color = glm::vec3(1.0f, 1.0f, 0.0f);
                    } else {
                      new_atom.vdw_radius = 1.2;
                      new_atom.element_color = glm::vec3(0.0f, 1.0f, 0.0f);
                      std::cout
                          << "WARNING: vdw radius of " << Atom.name
                          << " not known. Use 1.2 as default. Default color is green"
                          << std::endl;
                    }

                    this->atoms.push_back(new_atom);
                    this->numAtoms++;
                }
            }
        }
    } else {
        throw std::runtime_error("No models found in the CIF file " + filename);
    }
}

std::tuple<float, float, float, float, float, float>
Protein::ComputeBounds(int num_atoms) {
  // Find bounding box
  float xmin = this->atoms[0].coords.x;
  float xmax = xmin;
  float ymin = this->atoms[0].coords.y;
  float ymax = ymin;
  float zmin = this->atoms[0].coords.z;
  float zmax = zmin;
  for (int i = 1; i < num_atoms; ++i) {
    xmin = std::min(xmin, this->atoms[i].coords.x);
    xmax = std::max(xmax, this->atoms[i].coords.x);
    ymin = std::min(ymin, this->atoms[i].coords.y);
    ymax = std::max(ymax, this->atoms[i].coords.y);
    zmin = std::min(zmin, this->atoms[i].coords.z);
    zmax = std::max(zmax, this->atoms[i].coords.z);
  }
  return {xmin, xmax, ymin, ymax, zmin, zmax};
}

void Protein::ScaleBFactors() {
  auto [min_b, max_b] = std::minmax_element(
      this->atoms.begin(), this->atoms.end(),
      [](const atom a1, const atom a2) { return a1.b_factor < a2.b_factor; });
  float min = (*min_b).b_factor;
  float max = (*max_b).b_factor;
  for (auto &atom : atoms) {
    atom.b_factor -= min;
    atom.b_factor /= (max - min);
  }
}

float Protein::LargestVdwRadius() {
  float radius = 0;
  for (auto &atom : atoms) {
    radius = std::max(atom.vdw_radius, radius);
  }
  return radius;
}

#endif

