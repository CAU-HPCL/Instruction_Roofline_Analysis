#include "matrices.h"
#include <utility>



    // void CSR::read_from_mtx(std::ifstream &fin, Option option, bool zero_base)
    // {
    //     string line;
    //     while (getline(fin, line))
    //     {
    //         if (line[0] == '%')
    //             continue;
    //         else
    //         {
    //             stringstream sin_meta(line);
    //             sin_meta >> rows >> cols >> total_nonzeros;
    //             break;
    //         }
    //     }
    //     map<intT, vector<intT>> dict;
    //     map<intT, vector<DataT>> vals;

    //     intT tnz = 0;
    //     while (getline(fin, line))
    //     {
    //         intT r, c;
    //         DataT v = 1;
    //         stringstream sin(line);
    //         if (pattern_only)
    //             sin >> r >> c;
    //         else
    //             sin >> r >> c >> v;

    //         if (not zero_base)
    //         {
    //             r--;
    //             c--;
    //         }

    //         if (dict.find(r) == dict.end())
    //         {
    //             dict[r] = vector<intT>();
    //             vals[r] = vector<DataT>();
    //         }
    //         dict[r].push_back(c);
    //         if (not pattern_only)
    //             vals[r].push_back(v);
    //         tnz++;
    //     }

    //     assert(tnz == total_nonzeros);
    //     // zero padding
    //     original_rows = rows;
    //     original_cols = cols;

    //     if (option.zero_padding)
    //     {
    //         if (rows % option.block_size)
    //         {
    //             original_rows = rows;
    //             rows = ((rows - 1) / option.block_size + 1);
    //             //  rows = ((rows + option.block_size - 1) / option.block_size) * option.block_size;
    //         }

    //         if (cols % option.block_size)
    //         {
    //             original_cols = cols;
    //             cols = ((cols - 1) / option.block_size + 1);
    //             // cols = ((cols + option.block_size - 1) / option.block_size) * option.block_size;
    //         }
    //     }
    //     rowptr = new intT[rows + 1];
    //     // nzcount = new intT[rows];
    //     colidx = new intT[total_nonzeros];
    //     rowptr[0] = 0;
    //     if (not pattern_only)
    //     {
    //         values = new DataT[total_nonzeros];
    //     }
    //     intT offset = 0;
    //     for (int i = 0; i < rows; i++)
    //     {
    //         auto row_pos = dict[i];
    //         rowptr[i + 1] = rowptr[i] + row_pos.size();
    //         std::copy(dict[i].begin(), dict[i].end(), colidx + offset);

    //         if (not pattern_only) 
    //         {
    //             std::copy(vals[i].begin(), vals[i].end(), values + offset);
    //         }
    //         offset += row_pos.size();
    //     }
    // }

    static inline void strip_bom_and_ltrim(std::string& s) {
    if (s.size() >= 3 &&
        (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF) {
        s.erase(0, 3);
    }
    size_t p = 0;
    while (p < s.size() && std::isspace((unsigned char)s[p])) ++p;
    s.erase(0, p);
    }


    void CSR::read_from_mtx(std::ifstream &fin, Option option, bool zero_base)
    {
        bool coordinate = false;
        bool pattern = false;
        bool symetric = false;

        string line;
        if (!fin.is_open()) {
            std::cerr << "Failed to open  (" << std::strerror(errno) << ")\n";
            return; // 혹은 throw
        }
        if (!fin.good()) throw std::runtime_error("bad ifstream");
        fin.clear(); 
        fin.seekg(0, std::ios::beg);

        while (getline(fin, line))
        {   
            
            strip_bom_and_ltrim(line);
            if (line.empty()) continue;

            std::string lower = line;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

            if (lower[0] == '%')
            {
                if(lower.find("pattern") != string::npos)
                    pattern = true;
                if(lower.find("symmetric") != string::npos)
                    symetric = true;
                if(lower.find("coordinate") != string::npos)
                    coordinate = true;
            }
            else
            {
                stringstream sin_meta(lower);

                if(coordinate) 
                    sin_meta >> original_rows >> original_cols >> total_nonzeros;
                else
                {
                    sin_meta >> original_rows >> original_cols;
                    total_nonzeros = 0;
                }
                    
                
                break;
            }
        }

        pattern_only = (option.pattern_only && pattern);
        
        printf("Matrix info: %d rows, %d cols, %d nonzeros\n", original_rows, original_cols, total_nonzeros);
        printf("coordinate: %s\n", coordinate ? "true" : "false");
        printf("Matrix type: %s, %s\n", pattern ? "pattern" : "value", symetric ? "symmetric" : "asymmetric");



        if(option.zero_padding)
        {
            rows = (original_rows + option.block_size - 1) / option.block_size * option.block_size;
            cols = (original_cols + option.block_size - 1) / option.block_size * option.block_size;
        }else
        {
            rows = original_rows;
            cols = original_cols;
        }

        if(coordinate)
        {
            vector<vector<pair<uint32_t, DataT>>> row_datas(rows);
            int tnz = 0;
            int diag_count = 0;

            while(getline(fin, line))
            {
                uint32_t r, c;
                DataT v = 1;
                stringstream sin(line);

                if(pattern)
                    sin >> r >> c;
                else
                    sin >> r >> c >> v;

                if(zero_base)
                {
                    r--;
                    c--;
                }

                row_datas[r].emplace_back(make_pair(c, v));
                tnz++;
                if(symetric && r != c) {row_datas[c].emplace_back(make_pair(r, v));}
                if(r == c) diag_count++;

            } 

            rowptr = new intT[rows + 1]; 
            
            if(symetric)
                {colidx = new intT[(size_t)2 * tnz - diag_count]; values = new DataT[(size_t)2 * tnz - diag_count]; total_nonzeros = 2 * (size_t)tnz - diag_count; }
            else 
                {colidx = new intT[(size_t)tnz]; values = new DataT[(size_t)tnz];}

            rowptr[0] = 0;
            uint64_t offset = 0;
            for(uint64_t i = 1; i< rows + 1; i++)
            {
                auto &row_vec = row_datas[i-1];
                if(!row_vec.empty())
                {
                    sort(row_vec.begin(), row_vec.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });

                    row_vec.erase(std::unique(row_vec.begin(), row_vec.end(),
                    [](const auto& a, const auto& b){ return a.first == b.first; }),
                    row_vec.end());


                }
                rowptr[i] = rowptr[i-1] + row_vec.size();

                for(uint64_t j =0; j< row_vec.size(); j++)
                {
                    colidx[offset] = row_vec[j].first;
                    if(not pattern_only && not pattern)
                        values[offset] = row_vec[j].second;
                    else
                        values[offset] = 1;
                    offset++;
                }
            }

            // auto expected = symetric ? (2*tnz - diag_count) : tnz;
            // assert(rowptr[rows] == expected);

        }
        else
        {
            vector<vector<pair<intT, DataT>>> row_datas(rows);
            uint64_t index = 0;
            uint64_t diag_count = 0;
            uint64_t tnz = 0;

            while(getline(fin, line))
            {
                DataT v = 1;
                stringstream sin(line);
                sin >> v;

                if(v == 0) { index++; continue; } 

                uint64_t col_index = index / original_rows;
                uint64_t row_index = index % original_rows;

                row_datas[row_index].emplace_back(make_pair(col_index, v));
                tnz++;
                if(symetric && row_index != col_index) {row_datas[col_index].emplace_back(make_pair(row_index, v));}
                if(row_index == col_index) diag_count++;

                index++;
                total_nonzeros++;
            }

            rowptr = new intT[rows + 1];
            
            if(symetric)
                {colidx = new intT[(size_t)2 * tnz - diag_count]; values = new DataT[(size_t)2 * tnz - diag_count]; total_nonzeros = 2 * tnz - diag_count;}
            else
                {colidx = new intT[(size_t)tnz]; values = new DataT[(size_t)tnz];}


            // if(tnz != total_nonzeros)
            //     printf("Warning: total nonzeros mismatch, read %d, meta %d\n", tnz, total_nonzeros);
            // assert(tnz == total_nonzeros);


            rowptr[0] = 0;
            uint64_t offset = 0;
            for(uint64_t i = 1; i< rows + 1; i++)
            {
                auto &row_vec = row_datas[i-1];
                if(!row_vec.empty())
                {
                    sort(row_vec.begin(), row_vec.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });

                    row_vec.erase(std::unique(row_vec.begin(), row_vec.end(),
                    [](const auto& a, const auto& b){ return a.first == b.first; }),
                    row_vec.end());


                }
                rowptr[i] = rowptr[i-1] + row_vec.size();

                for(uint64_t j =0; j< row_vec.size(); j++)
                {
                    colidx[offset] = row_vec[j].first;
                    if(not pattern_only && not pattern)
                        values[offset] = row_vec[j].second;
                    else
                        values[offset] = 1;
                    offset++;
                }
            }

            // auto expected = symetric ? (2*tnz - diag_count) : tnz;
            // assert(rowptr[rows] == expected);
        }

    }



    void CSR::read_from_smtx(std::ifstream &fin, Option option, bool zero_base)
    {
        string line;
        string buffer;
        map<intT, vector<intT>> dict;
        // header
        getline(fin, line);
        stringstream sin_meta(line);
        getline(sin_meta, buffer, ',');
        rows = stoi(buffer);
        getline(sin_meta, buffer, ',');
        cols = stoi(buffer);
        getline(sin_meta, buffer, ',');
        total_nonzeros = stoi(buffer);
        // zero padding
        original_rows = rows;
        original_cols = cols;
        if (option.zero_padding) 
        {
            if (rows % option.block_size)
            {
                original_rows = rows;
                rows = ((rows - 1) / option.block_size + 1) * option.block_size;
            }
            if (cols % option.block_size)
            {
                original_cols = cols;
                cols = ((cols - 1) / option.block_size + 1) * option.block_size;
            }
        }
        rowptr = new intT[rows + 1];
        colidx = new intT[total_nonzeros];
        if (not pattern_only)
        {
            values = new DataT[total_nonzeros];
        }
        vector<intT> vec_colidx(total_nonzeros);
        // original_ja_contiguous = new intT[total_nonzeros];
        // vec_rowptr
        getline(fin, line);
        stringstream sin_row(line);
        intT offset = 0, idx = 0;
        while (getline(sin_row, buffer, ' '))
        {
            offset = stoi(buffer);
            rowptr[idx++] = offset;
            // vec_rowptr[idx++] = offset;
        }
        assert(idx == original_rows + 1);
        if (option.zero_padding)
        {
            for (int i = original_rows + 1; i <= rows; i++)
            {
                rowptr[i] = offset;
            }
        }
        int c;
        idx = 0;
        offset = 0;
        if (not pattern_only)
        {
            for (int i = 0; i < total_nonzeros; i++)
            {
                values[i] = 1;
            }
        }
        while (getline(fin, line))
        { 
            stringstream sin_col(line);
            while (getline(sin_col, buffer, ' '))
            {
                c = stoi(buffer);
                vec_colidx[offset++] = c;
            }
        }

        std::copy(vec_colidx.begin(), vec_colidx.end(), colidx);
        assert(offset == total_nonzeros);
        vec_colidx.clear();
    }


    void CSR_uint64_t::read_from_mtx_uint64_t(std::ifstream &fin, Option option, bool zero_base)
    {
        bool coordinate = false;
        bool pattern = false;
        bool symetric = false;

        string line;
        if (!fin.is_open()) {
            std::cerr << "Failed to open  (" << std::strerror(errno) << ")\n";
            return; // 혹은 throw
        }
        if (!fin.good()) throw std::runtime_error("bad ifstream");
        fin.clear(); 
        fin.seekg(0, std::ios::beg);

        while (getline(fin, line))
        {   
            
            strip_bom_and_ltrim(line);
            if (line.empty()) continue;

            std::string lower = line;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

            if (lower[0] == '%')
            {
                if(lower.find("pattern") != string::npos)
                    pattern = true;
                if(lower.find("symmetric") != string::npos)
                    symetric = true;
                if(lower.find("coordinate") != string::npos)
                    coordinate = true;
            }
            else
            {
                stringstream sin_meta(lower);

                if(coordinate) 
                    sin_meta >> original_rows >> original_cols >> total_nonzeros;
                else
                {
                    sin_meta >> original_rows >> original_cols;
                    total_nonzeros = 0;
                }
                    
                
                break;
            }
        }

        pattern_only = (option.pattern_only && pattern);
        

        if(option.zero_padding)
        {
            rows = (original_rows + option.block_size - 1) / option.block_size * option.block_size;
            cols = (original_cols + option.block_size - 1) / option.block_size * option.block_size;
        }else
        {
            rows = original_rows;
            cols = original_cols;
        }

        if(coordinate)
        {
            vector<vector<pair<uint32_t, DataT>>> row_datas(rows);
            uint64_t tnz = 0;
            uint64_t diag_count = 0;

            while(getline(fin, line))
            {
                uint32_t r, c;
                DataT v = 1;
                stringstream sin(line);

                if(pattern)
                    sin >> r >> c;
                else
                    sin >> r >> c >> v;

                if(zero_base)
                {
                    r--;
                    c--;
                }

                row_datas[r].emplace_back(make_pair(c, v));
                tnz++;
                if(symetric && r != c) {row_datas[c].emplace_back(make_pair(r, v));}
                if(r == c) diag_count++;

            } 

            rowptr = new uint64_t[rows + 1]; 
            
            if(symetric)
                {colidx = new uint64_t[(size_t)2 * tnz - diag_count]; values = new DataT[(size_t)2 * tnz - diag_count]; total_nonzeros = 2 * (size_t)tnz - diag_count; }
            else 
                {colidx = new uint64_t[(size_t)tnz]; values = new DataT[(size_t)tnz];}

            rowptr[0] = 0;
            uint64_t offset = 0;
            for(uint64_t i = 1; i< rows + 1; i++)
            {
                auto &row_vec = row_datas[i-1];
                if(!row_vec.empty())
                {
                    sort(row_vec.begin(), row_vec.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });

                    row_vec.erase(std::unique(row_vec.begin(), row_vec.end(),
                    [](const auto& a, const auto& b){ return a.first == b.first; }),
                    row_vec.end());


                }
                rowptr[i] = rowptr[i-1] + row_vec.size();

                for(uint64_t j =0; j< row_vec.size(); j++)
                {
                    colidx[offset] = row_vec[j].first;
                    if(not pattern_only && not pattern)
                        values[offset] = row_vec[j].second;
                    else
                        values[offset] = 1;
                    offset++;
                }
            }

            // auto expected = symetric ? (2*tnz - diag_count) : tnz;
            // assert(rowptr[rows] == expected);

        }
        else
        {
            vector<vector<pair<intT, DataT>>> row_datas(rows);
            uint64_t index = 0;
            uint64_t diag_count = 0;
            uint64_t tnz = 0;

            while(getline(fin, line))
            {
                DataT v = 1;
                stringstream sin(line);
                sin >> v;

                if(v == 0) { index++; continue; } 

                uint64_t col_index = index / original_rows;
                uint64_t row_index = index % original_rows;

                row_datas[row_index].emplace_back(make_pair(col_index, v));
                tnz++;
                if(symetric && row_index != col_index) {row_datas[col_index].emplace_back(make_pair(row_index, v));}
                if(row_index == col_index) diag_count++;

                index++;
                total_nonzeros++;
            }

            rowptr = new uint64_t[rows + 1];
            
            if(symetric)
                {colidx = new uint64_t[(size_t)2 * tnz - diag_count]; values = new DataT[(size_t)2 * tnz - diag_count]; total_nonzeros = 2 * tnz - diag_count;}
            else
                {colidx = new uint64_t[(size_t)tnz]; values = new DataT[(size_t)tnz];}


            // if(tnz != total_nonzeros)
            //     printf("Warning: total nonzeros mismatch, read %d, meta %d\n", tnz, total_nonzeros);
            // assert(tnz == total_nonzeros);


            rowptr[0] = 0;
            uint64_t offset = 0;
            for(uint64_t i = 1; i< rows + 1; i++)
            {
                auto &row_vec = row_datas[i-1];
                if(!row_vec.empty())
                {
                    sort(row_vec.begin(), row_vec.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });

                    row_vec.erase(std::unique(row_vec.begin(), row_vec.end(),
                    [](const auto& a, const auto& b){ return a.first == b.first; }),
                    row_vec.end());


                }
                rowptr[i] = rowptr[i-1] + row_vec.size();

                for(uint64_t j =0; j< row_vec.size(); j++)
                {
                    colidx[offset] = row_vec[j].first;
                    if(not pattern_only && not pattern)
                        values[offset] = row_vec[j].second;
                    else
                        values[offset] = 1;
                    offset++;
                }
            }

            // auto expected = symetric ? (2*tnz - diag_count) : tnz;
            // assert(rowptr[rows] == expected);
        }

    }

