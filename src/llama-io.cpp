#include "llama-io.h"

#include <vector>

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_write_i::write_tensor_strided(ggml_tensor * tensor, size_t offset, size_t row_size, size_t n_rows, size_t row_stride) {
    for (size_t i = 0; i < n_rows; ++i) {
        write_tensor(tensor, offset + i*row_stride, row_size);
    }
}

void llama_io_read_i::read_tensor_strided(ggml_tensor * tensor, size_t offset, size_t row_size, size_t n_rows, size_t row_stride) {
    for (size_t i = 0; i < n_rows; ++i) {
        read_tensor(tensor, offset + i*row_stride, row_size);
    }
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}
