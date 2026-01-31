#include "ColPackHeaders.h"

#include <string>
#include <vector>

extern "C" {

int colpack_color_graph(
    const int* row_offsets,
    const int* col_indices,
    int num_vertices,
    int num_edges,
    int* out_colors,
    int* out_color_count
) {
    if (!row_offsets || !col_indices || !out_colors || num_vertices < 0 || num_edges < 0) {
        return -1;
    }

    std::vector<unsigned int*> pattern(static_cast<size_t>(num_vertices));
    for (int i = 0; i < num_vertices; ++i) {
        int start = row_offsets[i];
        int end = row_offsets[i + 1];
        int degree = end - start;
        if (degree < 0) {
            return -2;
        }
        pattern[i] = new unsigned int[static_cast<size_t>(degree + 1)];
        pattern[i][0] = static_cast<unsigned int>(degree);
        for (int j = 0; j < degree; ++j) {
            pattern[i][j + 1] = static_cast<unsigned int>(col_indices[start + j]);
        }
    }

    ColPack::GraphColoringInterface coloring(SRC_MEM_ADOLC, pattern.data(), num_vertices);
    coloring.Coloring("LARGEST_FIRST", "DISTANCE_ONE");

    std::vector<int> colors;
    coloring.GetVertexColors(colors);
    if (static_cast<int>(colors.size()) != num_vertices) {
        for (int i = 0; i < num_vertices; ++i) {
            delete[] pattern[i];
        }
        return -3;
    }

    for (int i = 0; i < num_vertices; ++i) {
        out_colors[i] = colors[static_cast<size_t>(i)];
    }

    if (out_color_count) {
        *out_color_count = coloring.GetVertexColorCount();
    }

    for (int i = 0; i < num_vertices; ++i) {
        delete[] pattern[i];
    }

    return 0;
}

} // extern "C"
