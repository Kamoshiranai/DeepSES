
#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(r32f, binding = 0) uniform image3D img_dist_field;

struct atom {
  vec3 coords;
  float vdw_radius;
  vec3 element_color;
  float b_factor;
};
layout(std430, binding = 0) buffer atomsBuffer { atom atoms[]; };
layout(std430, binding = 1) buffer indexBuffer { uint index[]; };
layout(std430, binding = 2) buffer neighborsCounterBuffer { int n_counters[]; };
layout(std430, binding = 3) buffer neighborsBuffer { uint n_grid[]; };

// Resolution of the sdf grid
uniform float resolution;
uniform float r_probe;
// Maximum number of atoms in one cell of the neighborhood grid
uniform int cell_size;
// Size of the neighborhood grid
uniform vec3 n_dims; // x,y, and z size of the neighborhood grid (how many cells in each dim)
// Resolution of the neighborhood grid
uniform float n_resolution; // size in Angstrom?

atom closest;

// Distance to the surface of an atom
float sdAtom(vec3 pos, atom sphere) {
  return length(pos - sphere.coords) - (sphere.vdw_radius);
}

// Calculate the 1d index for the counter grid from the 3d indices
int counter_index(ivec3 grid_coords) {
  ivec3 xyz = ivec3(n_dims);
  return (grid_coords.z * xyz.y * xyz.x + grid_coords.y * xyz.x +
          grid_coords.x);
}

// Calculate the 1d index for the cells grid from the 3d index of the cell and
// the number of the atom in that cell
int cell_index(ivec3 grid_coords, int no_atom) {
  ivec3 xyz = ivec3(n_dims);
  return (grid_coords.z * xyz.y * xyz.x * cell_size +
          grid_coords.y * xyz.x * cell_size + grid_coords.x * cell_size +
          no_atom);
}

// Calculate the coords in the neighbor grid from the atom coords
// (molecule/global space)
ivec3 g_coords(vec3 atom_coords, vec3 n_dims, float resolution) {

  return ivec3(round((atom_coords / resolution) + (n_dims / 2.0f)));
}

void main() {
  // Dimensions of the sdf grid
  vec3 dims = imageSize(img_dist_field);
  // Coords in sdf grid space
  vec3 pixel_coords = vec3(gl_GlobalInvocationID.xyz);
  //  Calculate the position in molecule/global space
  vec3 pos = (pixel_coords - dims / 2.0f) * resolution;
  // Coords in neighbor grid space
  ivec3 grid_coords = g_coords(pos, n_dims, n_resolution);

  // Set minimum to some value larger than r_probe. This minimum will be updated
  // if atoms are sufficiently close
  float minimum = r_probe;

  // Iterate through neighboring cells in the neighbor grid
  for (int i = max(0, grid_coords.x - 1);
       i <= min(grid_coords.x + 1, n_dims.x - 1); ++i) {
    for (int j = max(0, grid_coords.y - 1);
         j <= min(grid_coords.y + 1, n_dims.y - 1); ++j) {
      for (int k = max(0, grid_coords.z - 1);
           k <= min(grid_coords.z + 1, n_dims.z - 1); ++k) {
        // How many neighbors are in this cell?
        int no_neighbors = n_counters[counter_index(ivec3(i, j, k))];
        // Iterate over every atom in this cell
        for (int l = 0; l < no_neighbors; ++l) {
          uint n_index = n_grid[cell_index(ivec3(i, j, k), l)];
          float dist = sdAtom(pos, atoms[n_index]);
          //  Save atom and dist, if this atom is closer than the previously
          //  visited atoms

          if (minimum > dist) {
            minimum = dist;
            closest = atoms[n_index];
          }
        }
      }
    }
  }

  // set sdf value inside to -grid distance = -resolution

  if (minimum < -1 * resolution) { //NOTE: new version
    minimum = -1 * resolution;
  }

  imageStore(img_dist_field, ivec3(pixel_coords), vec4(minimum, 0.0, 0.0, 0.0));
}