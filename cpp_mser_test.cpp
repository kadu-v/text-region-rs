// Standalone C++ MSER V1 implementation extracted from fast-mser.
// Uses OpenCV only for image I/O. Single-threaded path only.
// Faithfully follows img_fast_mser_v1.cpp algorithm.

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>

using std::vector;
using std::string;

// --- Type aliases matching C++ codebase ---
typedef uint8_t  u8;
typedef uint16_t u16;
typedef int16_t  i16;
typedef int32_t  i32;
typedef uint32_t u32;
typedef int64_t  i64;
typedef float    f32;

// --- Data structures ---

struct LinkedPoint {
    u16 m_x;
    u16 m_y;
    i32 m_next;
    i32 m_prev;
    i32 m_ref;
};

struct MserRegion {
    enum Flag { Flag_Unknow = 0, Flag_Invalid = 1, Flag_Valid = 2, Flag_Merged = 3 };

    u32 m_region_flag : 2;
    u32 m_gray_level : 8;
    u32 m_calculated_var : 1;
    u32 m_boundary_region : 1;
    u32 m_patch_index : 8;

    i32 m_size;
    union { u32 m_unmerged_size; f32 m_var; i32 m_mser_index; };

    MserRegion* m_parent;
    i32 m_head;
    i32 m_tail;
    u16 m_left, m_right, m_top, m_bottom;
};

struct ConnectedComp {
    i32 m_head;
    i32 m_tail;
    MserRegion* m_region;
    i16 m_gray_level;
    i32 m_size;
    u16 m_left, m_right, m_top, m_bottom;
};

// --- Block memory (simplified, no reuse) ---
template<class T>
class BlockMemory {
public:
    int block_size_2_base;
    int block_size;
    int block_size_mask;
    int cur_block;
    int element_number;
    vector<T*> datas;
    T* cur_data;
    T* cur_block_end;

    BlockMemory() : block_size_2_base(0), block_size(0), block_size_mask(0),
                    cur_block(-1), element_number(0), cur_data(nullptr), cur_block_end(nullptr) {}

    ~BlockMemory() { for (auto* d : datas) free(d); }

    void init(int base) {
        for (auto* d : datas) free(d);
        datas.clear();
        block_size_2_base = base;
        block_size = 1 << base;
        block_size_mask = block_size - 1;
        cur_block = -1;
        element_number = 0;
        next_block();
    }

    void next_block() {
        ++cur_block;
        if (cur_block > (int)datas.size() - 1) {
            T* d = (T*)malloc(sizeof(T) * block_size);
            datas.push_back(d);
            cur_data = d;
        } else {
            cur_data = datas[cur_block];
        }
        cur_block_end = cur_data + block_size;
    }

    T* get_next() { return cur_data; }
    void add() {
        ++cur_data;
        ++element_number;
        if (cur_data >= cur_block_end) next_block();
    }

    T& at(int index) { return datas[index >> block_size_2_base][index & block_size_mask]; }

    // Visitor
    struct Visitor {
        T* data = nullptr;
        int block_index = 0;
        int index = 0;
        int visit_number = 0;
    };

    bool visit(Visitor& v) {
        if (v.visit_number >= element_number) return false;
        if (v.index == block_size) { ++v.block_index; v.index = 0; }
        if (v.index == 0) v.data = datas[v.block_index]; else ++v.data;
        ++v.index;
        ++v.visit_number;
        return true;
    }
};

// --- MSER output ---
struct OutputMser {
    u8 gray_level;
    i32 size;
    vector<cv::Point> points;
    cv::Rect rect;
};

// --- MSER parameters ---
struct MserParams {
    int delta = 5;
    int min_point = 50;
    float max_point_ratio = 0.05f;
    float stable_variation = 0.25f;
    float duplicated_variation = 0.2f;
    float nms_similarity = 0.5f;
    bool from_min = true;
    bool from_max = true;
    // 4-connected only
};

// --- Helper macros ---
#define BOUNDARY_YES_MASK 0x4000

// --- Main MSER class ---
class FastMserV1 {
public:
    i16* masked_image = nullptr;
    LinkedPoint* linked_points = nullptr;
    i16** heap = nullptr;

    i32 width, height;
    i32 wb, hb; // with boundary

    BlockMemory<MserRegion> regions;
    i16** heap_start[257];
    ConnectedComp comp[256];
    i32 dir[8];
    u32 level_size[257];
    u32 dir_mask, dir_max;
    i32 er_number;
    LinkedPoint* linked_points_end;

    // Recognition state
    u32 region_level_size[257];
    u32 start_indexes[257];
    MserRegion** gray_order_regions = nullptr;
    u32 gray_order_region_size;

    MserParams params;
    i32 max_point;

    ~FastMserV1() {
        free(masked_image);
        free(linked_points);
        free(heap);
    }

    void allocate(int w, int h) {
        width = w; height = h;
        wb = w + 2; hb = h + 2;
        i32 pt_size = wb * hb;
        i32 h_size = w * h + 257;

        masked_image = (i16*)malloc(sizeof(i16) * pt_size);
        linked_points = (LinkedPoint*)malloc(sizeof(LinkedPoint) * pt_size);
        heap = (i16**)malloc(sizeof(i16*) * h_size);

        // 4-connected
        dir_max = 0x800;
        dir_mask = 0x0e00;
        dir[0] = 1;
        dir[1] = -wb;
        dir[2] = -1;
        dir[3] = wb;
    }

    void process_tree_patch(const u8* img_data, i32 img_stride, u8 gray_mask) {
        i16* md = masked_image;
        memset(level_size, 0, sizeof(u32) * 257);

        // Top border row
        for (int c = 0; c < wb; ++c) *md++ = -1;

        const u8* row_ptr = img_data;
        for (int r = 0; r < height; ++r) {
            *md++ = -1; // left border
            for (int c = 0; c < width; ++c) {
                u8 g = row_ptr[c] ^ gray_mask;
                level_size[g]++;
                *md++ = (i16)g;
            }
            *md++ = -1; // right border
            row_ptr += img_stride;
        }

        // Bottom border row
        for (int c = 0; c < wb; ++c) *md++ = -1;

        // Setup heap
        heap_start[0] = &heap[0];
        heap_start[0][0] = 0;
        for (int i = 1; i < 257; ++i) {
            heap_start[i] = heap_start[i-1] + level_size[i-1] + 1;
            heap_start[i][0] = 0;
        }
    }

    void init_comp(ConnectedComp* comptr, MserRegion* region, u8 patch_index) {
        comptr->m_size = 0;
        region->m_gray_level = (u8)comptr->m_gray_level;
        region->m_region_flag = MserRegion::Flag_Unknow;
        region->m_size = 0;
        region->m_unmerged_size = 0;
        region->m_parent = nullptr;
        region->m_calculated_var = 0;
        region->m_boundary_region = 0;
        region->m_patch_index = patch_index;
        comptr->m_region = region;
    }

    void new_region(ConnectedComp* comptr, MserRegion* region, u8 patch_index) {
        region->m_gray_level = (u8)comptr->m_gray_level;
        region->m_region_flag = MserRegion::Flag_Unknow;
        region->m_size = 0;
        region->m_unmerged_size = 0;
        region->m_parent = nullptr;
        region->m_calculated_var = 0;
        region->m_patch_index = patch_index;
        comptr->m_region = region;
    }

    void merge_comp(ConnectedComp* comp1, ConnectedComp* comp2) {
        comp1->m_region->m_parent = comp2->m_region;

        if (comp2->m_size > 0) {
            linked_points[comp2->m_tail].m_next = comp1->m_head;
            linked_points[comp1->m_head].m_prev = comp2->m_tail;
            comp2->m_tail = comp1->m_tail;

            if (comp1->m_left < comp2->m_left) comp2->m_left = comp1->m_left;
            if (comp1->m_right > comp2->m_right) comp2->m_right = comp1->m_right;
            if (comp1->m_top < comp2->m_top) comp2->m_top = comp1->m_top;
            if (comp1->m_bottom > comp2->m_bottom) comp2->m_bottom = comp1->m_bottom;
        } else {
            comp2->m_head = comp1->m_head;
            comp2->m_tail = comp1->m_tail;
            comp2->m_left = comp1->m_left;
            comp2->m_right = comp1->m_right;
            comp2->m_top = comp1->m_top;
            comp2->m_bottom = comp1->m_bottom;
        }

        comp2->m_size += comp1->m_size;
    }

    void make_tree_patch(const u8* img_data, i32 img_stride, u8 gray_mask) {
        process_tree_patch(img_data, img_stride, gray_mask);

        regions.init(11); // block_size = 2048

        i16*** heap_cur = heap_start;
        ConnectedComp* comptr = comp;
        i16* ioptr = masked_image + 1 + wb;
        i16* imgptr = ioptr;
        LinkedPoint* ptsptr = linked_points;

        comptr->m_gray_level = 257;
        comptr++;
        comptr->m_gray_level = (*imgptr) & 0x01ff;

        MserRegion* cur_region = regions.get_next();
        regions.add();
        init_comp(comptr, cur_region, 0);
        *imgptr |= (i16)0x8000;
        heap_cur += (*imgptr) & 0x01ff;

        i32 offset = 0;
        i32 pt_index;

        for (;;) {
            u16 dir_masked;
            while ((dir_masked = ((*imgptr) & dir_mask)) < dir_max) {
                i16* imgptr_nbr = imgptr + dir[dir_masked >> 9];

                if (*imgptr_nbr >= 0) {
                    *imgptr_nbr |= (i16)0x8000;
                    offset = ((*imgptr_nbr) & 0x01ff) - ((*imgptr) & 0x01ff);

                    if (offset < 0) {
                        (*heap_cur)++;
                        **heap_cur = imgptr;
                        *imgptr += 0x200;
                        heap_cur += offset;

                        imgptr = imgptr_nbr;
                        comptr++;
                        comptr->m_gray_level = (*imgptr) & 0x01ff;

                        cur_region = regions.get_next();
                        regions.add();
                        init_comp(comptr, cur_region, 0);
                        continue;
                    } else {
                        heap_cur[offset]++;
                        *heap_cur[offset] = imgptr_nbr;
                    }
                }
                *imgptr += 0x200;
            }

            // Record pixel coordinates
            i32 imsk = (i32)(imgptr - ioptr);
            ptsptr->m_y = imsk / wb;
            ptsptr->m_x = imsk - (ptsptr->m_y * wb);

            pt_index = (i32)(ptsptr - linked_points);

            if (comptr->m_size > 0) {
                ptsptr->m_next = comptr->m_head;
                linked_points[comptr->m_head].m_prev = pt_index;
                ptsptr->m_prev = -1;
                ptsptr->m_ref = -1;

                if (ptsptr->m_x < comptr->m_left) comptr->m_left = ptsptr->m_x;
                else if (ptsptr->m_x > comptr->m_right) comptr->m_right = ptsptr->m_x;
                if (ptsptr->m_y < comptr->m_top) comptr->m_top = ptsptr->m_y;
                else if (ptsptr->m_y > comptr->m_bottom) comptr->m_bottom = ptsptr->m_y;
            } else {
                ptsptr->m_prev = -1;
                ptsptr->m_next = -1;
                ptsptr->m_ref = -1;
                comptr->m_tail = pt_index;
                comptr->m_left = ptsptr->m_x;
                comptr->m_right = ptsptr->m_x;
                comptr->m_top = ptsptr->m_y;
                comptr->m_bottom = ptsptr->m_y;
            }

            comptr->m_head = pt_index;
            ++comptr->m_size;
            ++ptsptr;

            if (**heap_cur) {
                imgptr = **heap_cur;
                (*heap_cur)--;
            } else {
                // Finalize region
                comptr->m_region->m_head = comptr->m_head;
                comptr->m_region->m_tail = comptr->m_tail;
                comptr->m_region->m_size = comptr->m_size;
                comptr->m_region->m_unmerged_size = comptr->m_size;
                comptr->m_region->m_left = comptr->m_left;
                comptr->m_region->m_right = comptr->m_right;
                comptr->m_region->m_top = comptr->m_top;
                comptr->m_region->m_bottom = comptr->m_bottom;

                heap_cur++;
                i16 pixel_val = 0;
                for (i32 i = ((*imgptr) & 0x01ff) + 1; i < 257; ++i) {
                    if (**heap_cur) { pixel_val = i; break; }
                    heap_cur++;
                }

                if (pixel_val) {
                    imgptr = **heap_cur;
                    (*heap_cur)--;

                    if (pixel_val < comptr[-1].m_gray_level) {
                        cur_region = regions.get_next();
                        regions.add();
                        if (comptr->m_region != cur_region) {
                            comptr->m_region->m_parent = cur_region;
                            cur_region->m_boundary_region = comptr->m_region->m_boundary_region;
                        }
                        comptr[0].m_gray_level = pixel_val;
                        new_region(comptr, cur_region, 0);
                    } else {
                        for (;;) {
                            comptr--;
                            merge_comp(comptr + 1, comptr);
                            if (pixel_val <= comptr[0].m_gray_level) break;

                            if (pixel_val < comptr[-1].m_gray_level) {
                                cur_region = regions.get_next();
                                regions.add();
                                if (comptr->m_region != cur_region) {
                                    comptr->m_region->m_parent = cur_region;
                                    cur_region->m_boundary_region = comptr->m_region->m_boundary_region;
                                }
                                comptr[0].m_gray_level = pixel_val;
                                new_region(comptr, cur_region, 0);
                                break;
                            }
                        }
                    }
                } else {
                    break;
                }
            }
        }

        er_number = regions.element_number;
        linked_points_end = ptsptr;
    }

    void recognize_mser_parallel_worker() {
        typename BlockMemory<MserRegion>::Visitor bp;

        while (regions.visit(bp)) {
            if (bp.data->m_region_flag == MserRegion::Flag_Merged) continue;

            i32 gray_level_threshold = bp.data->m_gray_level + params.delta;
            MserRegion* start_region = bp.data;
            MserRegion* parent_region = start_region->m_parent;
            // get_set_real_parent_for_merged
            while (parent_region != nullptr && parent_region->m_region_flag == MserRegion::Flag_Merged)
                parent_region = parent_region->m_parent;
            start_region->m_parent = parent_region;

            while (parent_region != nullptr && (i32)parent_region->m_gray_level <= gray_level_threshold) {
                start_region = parent_region;
                // get_set_real_parent_for_merged
                MserRegion* temp = parent_region->m_parent;
                while (temp != nullptr && temp->m_region_flag == MserRegion::Flag_Merged)
                    temp = temp->m_parent;
                parent_region->m_parent = temp;
                parent_region = temp;
            }

            if (parent_region != nullptr || (i32)start_region->m_gray_level == gray_level_threshold) {
                bp.data->m_var = (start_region->m_size - bp.data->m_size) / (f32)bp.data->m_size;
            } else {
                bp.data->m_var = -1;
            }

            if (bp.data->m_var > params.stable_variation) {
                bp.data->m_region_flag = MserRegion::Flag_Invalid;
            } else if (bp.data->m_size < params.min_point || bp.data->m_size > max_point || bp.data->m_parent == nullptr) {
                bp.data->m_region_flag = MserRegion::Flag_Invalid;
            }
        }
    }

    void get_duplicated_regions(vector<MserRegion*>& duplicated, MserRegion* stable, MserRegion* begin) {
        MserRegion* parent = begin->m_parent;
        while (true) {
            if (parent == nullptr) break;
            if (parent->m_size > max_point) break;
            double var = (parent->m_size - stable->m_size) / (double)stable->m_size;
            if (var > params.duplicated_variation) break;
            if (parent->m_region_flag == MserRegion::Flag_Invalid) {
                parent = parent->m_parent;
                continue;
            }
            duplicated.push_back(parent);
            parent = parent->m_parent;
        }
    }

    void recognize_mser() {
        recognize_mser_parallel_worker();

        // NMS + counting
        MserRegion* parent_region;
        memset(region_level_size, 0, sizeof(u32) * 257);
        u32 totalUnknownSize = 0;

        {
            typename BlockMemory<MserRegion>::Visitor bp;
            while (regions.visit(bp)) {
                if (bp.data->m_region_flag == MserRegion::Flag_Merged) continue;

                parent_region = bp.data->m_parent;

                if (bp.data->m_region_flag == MserRegion::Flag_Unknow) {
                    region_level_size[bp.data->m_gray_level]++;
                    totalUnknownSize++;
                    bp.data->m_calculated_var = 1;
                } else if (parent_region == nullptr || parent_region->m_region_flag == MserRegion::Flag_Invalid) {
                    continue;
                }

                if (parent_region != nullptr && parent_region->m_region_flag != MserRegion::Flag_Merged) {
                    if (params.nms_similarity >= 0 && bp.data->m_var >= 0 && parent_region->m_var >= 0
                        && (int)parent_region->m_gray_level == (int)bp.data->m_gray_level + 1) {
                        double subValue = parent_region->m_var - bp.data->m_var;
                        if (subValue > params.nms_similarity) {
                            if (MserRegion::Flag_Unknow == parent_region->m_region_flag) {
                                if (parent_region->m_calculated_var == 1) {
                                    --region_level_size[parent_region->m_gray_level];
                                    --totalUnknownSize;
                                }
                                parent_region->m_region_flag = MserRegion::Flag_Invalid;
                            }
                        } else if (-subValue > params.nms_similarity) {
                            if (MserRegion::Flag_Unknow == bp.data->m_region_flag) {
                                if (bp.data->m_calculated_var == 1) {
                                    --region_level_size[bp.data->m_gray_level];
                                    --totalUnknownSize;
                                }
                                bp.data->m_region_flag = MserRegion::Flag_Invalid;
                            }
                        }
                    }
                }
            }
        }

        // Gray-order sort
        // integral array
        start_indexes[0] = 0;
        for (int i = 0; i < 256; ++i) {
            start_indexes[i + 1] = start_indexes[i] + region_level_size[i];
        }

        gray_order_region_size = totalUnknownSize;
        gray_order_regions = (MserRegion**)heap;

        {
            typename BlockMemory<MserRegion>::Visitor bp;
            while (regions.visit(bp)) {
                if (bp.data->m_region_flag == MserRegion::Flag_Unknow) {
                    gray_order_regions[start_indexes[bp.data->m_gray_level]++] = bp.data;
                }
            }
        }

        // Duplicate removal
        if (params.duplicated_variation > 0) {
            u32 validCount = 0;
            vector<MserRegion*> helper;
            helper.reserve(100);

            for (u32 i = 0; i < gray_order_region_size; ++i) {
                MserRegion* cur = gray_order_regions[i];
                if (cur->m_region_flag != MserRegion::Flag_Unknow) continue;

                helper.clear();
                helper.push_back(cur);
                get_duplicated_regions(helper, cur, cur);

                i32 middleIndex = (i32)helper.size() / 2;

                if (middleIndex > 0) {
                    get_duplicated_regions(helper, helper[middleIndex], helper.back());
                }

                for (i32 j = 0; j < (i32)helper.size(); ++j) {
                    if (j != middleIndex) {
                        helper[j]->m_region_flag = MserRegion::Flag_Invalid;
                    } else {
                        helper[j]->m_region_flag = MserRegion::Flag_Valid;
                        ++validCount;
                    }
                }
            }

            // Compact
            u32 index = 0;
            for (u32 i = 0; i < totalUnknownSize; ++i) {
                if (gray_order_regions[i]->m_region_flag == MserRegion::Flag_Valid) {
                    gray_order_regions[index++] = gray_order_regions[i];
                }
            }
            gray_order_region_size = validCount;
        } else {
            for (u32 i = 0; i < gray_order_region_size; ++i) {
                gray_order_regions[i]->m_region_flag = MserRegion::Flag_Valid;
            }
        }
    }

    void extract_pixel(vector<OutputMser>& output, u8 gray_mask) {
        if (gray_order_region_size == 0) return;

        // Find top regions (whose parent is not Valid)
        vector<MserRegion*> top_regions;
        for (u32 i = 0; i < gray_order_region_size; ++i) {
            MserRegion* cur = gray_order_regions[i];
            MserRegion* real_parent = cur->m_parent;
            while (real_parent != nullptr && real_parent->m_region_flag != MserRegion::Flag_Valid)
                real_parent = real_parent->m_parent;

            if (real_parent == nullptr) {
                top_regions.push_back(cur);
            }
        }

        // Extract pixels from top regions via linked list, set m_ref
        vector<cv::Point> all_points;
        for (auto* cur : top_regions) {
            i32 pt_index = cur->m_head;
            for (i32 j = 0; j < cur->m_size; ++j) {
                LinkedPoint& lpt = linked_points[pt_index];
                lpt.m_ref = (i32)all_points.size();
                all_points.push_back(cv::Point(lpt.m_x, lpt.m_y));
                pt_index = lpt.m_next;
            }
        }

        // Build output MSERs
        for (u32 i = 0; i < gray_order_region_size; ++i) {
            MserRegion* cur = gray_order_regions[i];
            OutputMser mser;
            mser.gray_level = cur->m_gray_level ^ gray_mask;
            mser.size = cur->m_size;

            // Follow m_ref to find real tail
            i32 head_ref = linked_points[cur->m_head].m_ref;
            i32 tail_ref = linked_points[cur->m_tail].m_ref;
            // For tail, follow m_ref chain
            i32 real_tail = cur->m_tail;
            while (linked_points[real_tail].m_ref != -1) {
                // Actually in parallel_1_fast, m_ref is set to memory offset
                // For the 1-thread path, head_ref and tail_ref give the range
                break;
            }

            i32 start = std::min(head_ref, tail_ref);
            i32 end = std::max(head_ref, tail_ref);
            for (i32 j = start; j <= end; ++j) {
                mser.points.push_back(all_points[j]);
            }

            // Compute bounding rect
            int minX = INT_MAX, maxX = INT_MIN, minY = INT_MAX, maxY = INT_MIN;
            for (auto& p : mser.points) {
                if (p.x < minX) minX = p.x;
                if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }
            mser.rect = cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);

            output.push_back(std::move(mser));
        }
    }

    void extract(const u8* img_data, i32 w, i32 h, i32 stride,
                 vector<OutputMser>& from_min_out, vector<OutputMser>& from_max_out) {
        max_point = (i32)(params.max_point_ratio * w * h);
        allocate(w, h);

        if (params.from_min) {
            make_tree_patch(img_data, stride, 0);
            recognize_mser();
            extract_pixel(from_min_out, 0);
        }

        if (params.from_max) {
            make_tree_patch(img_data, stride, 255);
            recognize_mser();
            extract_pixel(from_max_out, 255);
        }
    }
};

// --- Drawing functions ---
void draw_rect(cv::Mat& img, const cv::Rect& r, cv::Scalar color) {
    cv::rectangle(img, r, color, 1);
}

void draw_pixels(cv::Mat& img, const vector<cv::Point>& pts, cv::Scalar color) {
    for (auto& p : pts) {
        if (p.x >= 0 && p.x < img.cols && p.y >= 0 && p.y < img.rows) {
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b((u8)color[0], (u8)color[1], (u8)color[2]);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    string path = argv[1];
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return 1;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    int w = gray.cols, h = gray.rows;

    float total_pixels = (float)(w * h);
    int min_point = std::max((int)(total_pixels * 0.0001f), 50);

    FastMserV1 mser;
    mser.params.delta = 5;
    mser.params.min_point = min_point;
    mser.params.max_point_ratio = 0.05f;
    mser.params.stable_variation = 0.25f;
    mser.params.duplicated_variation = 0.2f;
    mser.params.nms_similarity = 0.5f;
    mser.params.from_min = true;
    mser.params.from_max = true;

    std::cerr << "Detecting MSERs (C++ V1, 4-connected) on " << w << "x" << h << " image: " << path << std::endl;

    vector<OutputMser> from_min, from_max;
    mser.extract(gray.data, w, h, (i32)gray.step[0], from_min, from_max);

    std::cerr << "  from_min: " << from_min.size() << " regions, from_max: " << from_max.size() << " regions" << std::endl;

    // --- Bounding box visualization ---
    cv::Mat bbox_img = img.clone();
    for (auto& m : from_min) draw_rect(bbox_img, m.rect, cv::Scalar(0, 0, 255)); // red in BGR
    for (auto& m : from_max) draw_rect(bbox_img, m.rect, cv::Scalar(255, 0, 0)); // blue in BGR

    string stem = path;
    auto dot = stem.rfind('.');
    if (dot != string::npos) stem = stem.substr(0, dot);
    string bbox_path = stem + "_cpp_bbox.png";
    cv::imwrite(bbox_path, bbox_img);
    std::cerr << "  Saved bounding boxes: " << bbox_path << std::endl;

    // --- Pixel visualization ---
    cv::Mat pixel_img(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            u8 v = gray.at<u8>(y, x);
            pixel_img.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
        }

    for (auto& m : from_min) draw_pixels(pixel_img, m.points, cv::Scalar(50, 50, 255)); // red-ish in BGR
    for (auto& m : from_max) draw_pixels(pixel_img, m.points, cv::Scalar(255, 50, 50)); // blue-ish in BGR

    string pixel_path = stem + "_cpp_pixels.png";
    cv::imwrite(pixel_path, pixel_img);
    std::cerr << "  Saved pixel regions: " << pixel_path << std::endl;

    return 0;
}
