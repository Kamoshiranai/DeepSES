
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; //this was also local_size_y = 1

struct atom {
  vec3 coords;
  float vdw_radius;
  vec3 element_color;
  float b_factor;
};

layout(std430, binding = 0) buffer atomsBuffer { atom atoms[]; };
layout(std430, binding = 2) buffer gridCountersBuffer { int counters[]; };
layout(std430, binding = 3) buffer gridCellsBuffer { uint cells[]; };

uniform vec3 n_dims;
uniform float n_resolution;
// number of atoms that can be stored in a single cell
uniform int cell_size;
uniform float time;

// From atom_coords (coords as written in the atoms buffer, global space) to
// neighborhood grid space (cells and counters)
ivec3 g_coords(vec3 atom_coords, vec3 n_dims, float resolution) {

  return ivec3(round((atom_coords / resolution) + (n_dims / 2.0f)));
}

// Calculate 1D index from 3d index for the counters array
int counter_index(ivec3 grid_coords) {
  ivec3 xyz = ivec3(n_dims);
  return (grid_coords.z * xyz.y * xyz.x + grid_coords.y * xyz.x +
          grid_coords.x);
}

// Calculate 1D index from 3d index and the index of the atom in the cell, for
// the cells array
int cell_index(ivec3 grid_coords, int no_atom) {
  ivec3 xyz = ivec3(n_dims);
  return (grid_coords.z * xyz.y * xyz.x * cell_size +
          grid_coords.y * xyz.x * cell_size + grid_coords.x * cell_size +
          no_atom);
}

void main() {
  atom cur_atom = atoms[gl_GlobalInvocationID.x];

  // Find grid coordinates from atom coordinates
  ivec3 grid_coords = g_coords(cur_atom.coords, n_dims, n_resolution);

  // Calculate 1d index from 3d coords
  int counter_ind = counter_index(grid_coords);

  // Atomic add to counter of that cell
  int no_atom = atomicAdd(counters[counter_ind], 1);

  // Write atom id into cell
  int cell_ind = cell_index(grid_coords, no_atom);
  cells[cell_ind] = gl_GlobalInvocationID.x;
}
