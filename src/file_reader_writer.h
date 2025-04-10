
#pragma once
#include <fstream>
#include <vector>

#include "linalg.h"

namespace network {

class FileWriter {
public:
    explicit FileWriter(const std::string& filename) : file_(filename, std::ios::binary) {
    }

    template <typename T>
    typename std::enable_if_t<std::is_arithmetic_v<T>, FileWriter&> operator<<(const T& value) {
        file_.write(reinterpret_cast<const char*>(&value), sizeof(T));
        return *this;
    }

    template <typename T, typename A>
    FileWriter& operator<<(const std::vector<T, A>& vec) {
        size_t size = vec.size();
        *this << size;
        for (const auto& element : vec) {
            *this << element;
        }
        return *this;
    }

    FileWriter& operator<<(const Matrix& mat) {
        Matrix::Index rows = mat.rows();
        Matrix::Index cols = mat.cols();
        file_.write(reinterpret_cast<const char*>(&rows), sizeof(Index));
        file_.write(reinterpret_cast<const char*>(&cols), sizeof(Index));
        file_.write(reinterpret_cast<const char*>(mat.data()), rows * cols * sizeof(double));
        return *this;
    }

    FileWriter& operator<<(const Vector& vec) {
        Vector::Index size = vec.size();
        file_.write(reinterpret_cast<const char*>(&size), sizeof(Index));
        file_.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(double));
        return *this;
    }
    void CloseFile() {
        file_.close();
    }

private:
    std::ofstream file_;
};

class FileReader {
public:
    explicit FileReader(const std::string& filename) : file_(filename, std::ios::binary) {
    }

    template <typename T>
    typename std::enable_if_t<std::is_arithmetic_v<T>, FileReader&> operator>>(T& value) {
        file_.read(reinterpret_cast<char*>(&value), sizeof(T));
        return *this;
    }

    template <typename T, typename A>
    FileReader& operator>>(std::vector<T, A>& vec) {
        vec.clear();
        size_t size;
        *this >> size;
        vec.reserve(size);
        T element;
        for (size_t i = 0; i < size; ++i) {

            *this >> element;
            vec.push_back(std::move(element));
        }

        return *this;
    }

    FileReader& operator>>(Matrix& mat) {
        Index rows, cols;
        file_.read(reinterpret_cast<char*>(&rows), sizeof(Index));
        file_.read(reinterpret_cast<char*>(&cols), sizeof(Index));
        mat.resize(rows, cols);
        file_.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));
        return *this;
    }

    FileReader& operator>>(Vector& vec) {
        Index size;
        file_.read(reinterpret_cast<char*>(&size), sizeof(Index));
        vec.resize(size);
        file_.read(reinterpret_cast<char*>(vec.data()), size * sizeof(double));
        return *this;
    }
    void CloseFile() {
        file_.close();
    }

private:
    std::ifstream file_;
};

}  // namespace network