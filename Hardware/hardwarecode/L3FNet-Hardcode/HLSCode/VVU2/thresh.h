static ThresholdsActivation<20,9,15,ap_int<11>,ap_uint<4>,0,comp::less_equal<ap_int<11>, ap_int<11>>> threshs                     = {{{{18, 28, 38, 47, 57, 67, 76, 86, 95, 105, 115, 124, 134, 143, 153},
   {-3, 5, 13, 21, 28, 36, 44, 52, 59, 67, 75, 83, 90, 98, 106},
   {-17, -4, 8, 21, 34, 46, 59, 72, 84, 97, 110, 122, 135, 148, 160},
   {6, 16, 26, 35, 45, 55, 64, 74, 84, 93, 103, 112, 122, 132, 141},
   {2, 11, 21, 31, 41, 51, 61, 71, 81, 90, 100, 110, 120, 130, 140},
   {-10, 0, 9, 19, 28, 38, 47, 57, 66, 76, 85, 95, 104, 114, 123},
   {-19, -6, 7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163},
   {-4, 7, 18, 29, 40, 51, 62, 73, 84, 95, 106, 117, 128, 139, 150},
   {-17, -5, 7, 19, 31, 43, 56, 68, 80, 92, 104, 116, 128, 140, 153},
   {13, 25, 37, 48, 60, 71, 83, 94, 106, 117, 129, 140, 152, 163, 175},
   {14, 25, 35, 46, 57, 67, 78, 89, 99, 110, 121, 131, 142, 153, 163},
   {22, 36, 50, 63, 77, 91, 104, 118, 132, 145, 159, 173, 186, 200, 213},
   {14, 27, 40, 53, 66, 79, 92, 105, 118, 130, 143, 156, 169, 182, 195},
   {-12, 0, 12, 23, 35, 46, 58, 69, 81, 93, 104, 116, 127, 139, 150},
   {-6, 5, 17, 28, 39, 51, 62, 73, 84, 96, 107, 118, 130, 141, 152},
   {17, 26, 36, 45, 55, 64, 74, 83, 92, 102, 111, 121, 130, 140, 149},
   {-19, -7, 6, 18, 31, 43, 56, 68, 81, 93, 105, 118, 130, 143, 155},
   {-6, 3, 13, 23, 33, 43, 53, 62, 72, 82, 92, 102, 111, 121, 131},
   {7, 15, 23, 31, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 121},
   {13, 26, 38, 50, 62, 74, 87, 99, 111, 123, 136, 148, 160, 172, 184}},

  {{13, 20, 27, 34, 40, 47, 54, 61, 67, 74, 81, 87, 94, 101, 108},
   {-8, 1, 10, 19, 29, 38, 47, 56, 65, 74, 84, 93, 102, 111, 120},
   {-10, 4, 17, 30, 44, 57, 70, 84, 97, 110, 124, 137, 150, 164, 177},
   {35, 50, 66, 82, 98, 113, 129, 145, 161, 176, 192, 208, 224, 239,
    255},
   {42, 53, 65, 76, 88, 99, 110, 122, 133, 145, 156, 167, 179, 190, 202},
   {21, 28, 35, 42, 50, 57, 64, 71, 78, 85, 92, 99, 106, 113, 121},
   {7, 16, 26, 35, 45, 54, 63, 73, 82, 92, 101, 111, 120, 129, 139},
   {21, 31, 42, 52, 62, 73, 83, 94, 104, 114, 125, 135, 145, 156, 166},
   {6, 15, 23, 32, 40, 49, 58, 66, 75, 83, 92, 100, 109, 117, 126},
   {-22, -8, 6, 20, 34, 48, 62, 75, 89, 103, 117, 131, 145, 159, 172},
   {-10, 1, 11, 21, 31, 41, 51, 61, 71, 82, 92, 102, 112, 122, 132},
   {21, 30, 39, 48, 57, 66, 75, 84, 94, 103, 112, 121, 130, 139, 148},
   {6, 14, 22, 30, 37, 45, 53, 61, 69, 77, 84, 92, 100, 108, 116},
   {-10, -1, 8, 17, 26, 36, 45, 54, 63, 72, 81, 91, 100, 109, 118},
   {17, 24, 31, 38, 45, 53, 60, 67, 74, 81, 88, 95, 102, 109, 117},
   {20, 32, 45, 58, 70, 83, 95, 108, 121, 133, 146, 158, 171, 184, 196},
   {-8, 3, 14, 25, 36, 47, 58, 69, 80, 91, 102, 113, 123, 134, 145},
   {-13, -2, 10, 21, 33, 45, 56, 68, 79, 91, 102, 114, 126, 137, 149},
   {-28, -18, -9, 1, 10, 19, 29, 38, 47, 57, 66, 76, 85, 94, 104},
   {9, 15, 21, 27, 32, 38, 44, 49, 55, 61, 66, 72, 78, 84, 89}},

  {{21, 33, 44, 56, 67, 79, 90, 102, 113, 125, 136, 148, 159, 170, 182},
   {-14, -1, 11, 24, 36, 49, 61, 74, 86, 99, 111, 124, 136, 149, 161},
   {-20, -6, 9, 24, 39, 54, 69, 83, 98, 113, 128, 143, 158, 172, 187},
   {6, 16, 25, 34, 43, 52, 62, 71, 80, 89, 98, 108, 117, 126, 135},
   {7, 15, 24, 33, 41, 50, 59, 67, 76, 85, 93, 102, 111, 119, 128},
   {27, 39, 51, 63, 75, 87, 99, 111, 123, 135, 147, 159, 171, 183, 195},
   {15, 26, 36, 47, 58, 68, 79, 89, 100, 110, 121, 131, 142, 152, 163},
   {-22, -11, 1, 12, 24, 35, 46, 58, 69, 81, 92, 104, 115, 126, 138},
   {48, 61, 75, 88, 101, 115, 128, 141, 155, 168, 181, 195, 208, 221,
    235},
   {22, 34, 46, 57, 69, 81, 93, 105, 117, 129, 141, 153, 165, 177, 188},
   {-2, 8, 18, 28, 38, 49, 59, 69, 79, 89, 99, 109, 120, 130, 140},
   {7, 15, 23, 32, 40, 48, 57, 65, 74, 82, 90, 99, 107, 115, 124},
   {-34, -19, -5, 10, 25, 39, 54, 68, 83, 98, 112, 127, 141, 156, 171},
   {13, 25, 36, 48, 59, 71, 82, 94, 105, 117, 128, 140, 151, 163, 174},
   {79, 106, 133, 160, 188, 215, 242, 269, 296, 323, 350, 377, 405, 432,
    459},
   {43, 58, 74, 90, 106, 122, 138, 154, 169, 185, 201, 217, 233, 249,
    265},
   {15, 26, 36, 46, 56, 66, 76, 86, 96, 107, 117, 127, 137, 147, 157},
   {8, 17, 26, 35, 44, 53, 62, 71, 80, 89, 98, 107, 116, 125, 134},
   {4, 17, 30, 44, 57, 70, 83, 96, 109, 122, 135, 148, 162, 175, 188},
   {-37, -23, -9, 5, 19, 33, 47, 61, 75, 89, 103, 117, 131, 145, 160}},

  {{29, 43, 57, 71, 86, 100, 114, 128, 142, 156, 170, 185, 199, 213,
    227},
   {28, 39, 51, 63, 74, 86, 98, 109, 121, 133, 144, 156, 168, 179, 191},
   {-2, 8, 19, 29, 39, 49, 59, 70, 80, 90, 100, 110, 121, 131, 141},
   {59, 78, 98, 117, 137, 156, 176, 195, 215, 234, 253, 273, 292, 312,
    331},
   {50, 66, 81, 97, 113, 128, 144, 159, 175, 191, 206, 222, 237, 253,
    268},
   {11, 20, 29, 39, 48, 58, 67, 76, 86, 95, 105, 114, 124, 133, 142},
   {11, 20, 30, 39, 49, 58, 68, 77, 87, 96, 106, 115, 125, 134, 144},
   {42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174, 186, 198,
    210},
   {-31, -16, 0, 16, 31, 47, 62, 78, 94, 109, 125, 141, 156, 172, 188},
   {13, 21, 29, 37, 45, 53, 61, 70, 78, 86, 94, 102, 110, 118, 126},
   {-35, -23, -11, 2, 14, 26, 38, 51, 63, 75, 87, 100, 112, 124, 136},
   {-49, -34, -19, -4, 10, 25, 40, 55, 69, 84, 99, 114, 128, 143, 158},
   {-13, -2, 9, 21, 32, 43, 54, 66, 77, 88, 99, 110, 122, 133, 144},
   {-29, -18, -7, 5, 16, 27, 38, 49, 60, 71, 82, 93, 104, 116, 127},
   {-8, 0, 8, 15, 23, 31, 38, 46, 54, 61, 69, 77, 84, 92, 100},
   {-6, 3, 12, 22, 31, 40, 50, 59, 69, 78, 87, 97, 106, 115, 125},
   {-2, 11, 23, 36, 49, 61, 74, 86, 99, 111, 124, 136, 149, 161, 174},
   {-14, -3, 8, 19, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129, 140},
   {-28, -12, 5, 21, 38, 54, 70, 87, 103, 119, 136, 152, 168, 185, 201},
   {37, 55, 73, 90, 108, 126, 144, 161, 179, 197, 215, 232, 250, 268,
    286}},

  {{0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126},
   {4, 18, 32, 46, 60, 74, 88, 101, 115, 129, 143, 157, 171, 185, 199},
   {20, 29, 38, 47, 55, 64, 73, 82, 91, 99, 108, 117, 126, 135, 143},
   {40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248,
    264},
   {-18, -4, 9, 22, 36, 49, 63, 76, 90, 103, 117, 130, 144, 157, 170},
   {7, 18, 29, 41, 52, 63, 74, 85, 96, 107, 118, 129, 140, 151, 163},
   {13, 21, 29, 36, 44, 52, 59, 67, 74, 82, 90, 97, 105, 113, 120},
   {2, 17, 31, 46, 60, 75, 89, 103, 118, 132, 147, 161, 176, 190, 205},
   {-5, 9, 23, 37, 51, 65, 79, 93, 107, 121, 135, 149, 163, 177, 191},
   {14, 25, 35, 46, 56, 67, 77, 88, 98, 108, 119, 129, 140, 150, 161},
   {38, 51, 65, 78, 92, 106, 119, 133, 146, 160, 173, 187, 200, 214,
    227},
   {2, 15, 27, 39, 51, 63, 75, 88, 100, 112, 124, 136, 148, 161, 173},
   {-14, -2, 10, 22, 34, 46, 58, 70, 82, 94, 106, 118, 130, 142, 154},
   {-29, -14, 1, 16, 32, 47, 62, 78, 93, 108, 124, 139, 154, 170, 185},
   {-2, 9, 20, 30, 41, 52, 63, 74, 85, 95, 106, 117, 128, 139, 150},
   {-45, -30, -16, -2, 12, 27, 41, 55, 70, 84, 98, 113, 127, 141, 156},
   {-32, -19, -7, 6, 18, 31, 44, 56, 69, 81, 94, 106, 119, 132, 144},
   {-18, -8, 3, 14, 25, 36, 46, 57, 68, 79, 90, 100, 111, 122, 133},
   {-27, -14, 0, 13, 26, 40, 53, 67, 80, 93, 107, 120, 134, 147, 160},
   {9, 17, 26, 34, 43, 52, 60, 69, 77, 86, 94, 103, 111, 120, 129}},

  {{-16, -6, 3, 12, 21, 30, 39, 48, 57, 66, 75, 84, 93, 102, 112},
   {16, 26, 35, 44, 53, 63, 72, 81, 90, 99, 109, 118, 127, 136, 145},
   {2, 9, 17, 25, 33, 41, 49, 57, 65, 73, 80, 88, 96, 104, 112},
   {9, 21, 33, 45, 57, 69, 81, 93, 106, 118, 130, 142, 154, 166, 178},
   {-2, 5, 12, 19, 25, 32, 39, 46, 53, 59, 66, 73, 80, 87, 93},
   {36, 49, 63, 77, 90, 104, 118, 131, 145, 158, 172, 186, 199, 213,
    227},
   {12, 26, 40, 54, 68, 82, 96, 110, 124, 138, 152, 166, 180, 194, 208},
   {14, 23, 32, 41, 50, 59, 68, 78, 87, 96, 105, 114, 123, 132, 141},
   {-33, -18, -4, 10, 25, 39, 54, 68, 82, 97, 111, 126, 140, 154, 169},
   {-5, 8, 20, 32, 44, 56, 69, 81, 93, 105, 117, 130, 142, 154, 166},
   {-22, -6, 9, 25, 40, 56, 71, 86, 102, 117, 133, 148, 164, 179, 194},
   {-10, -1, 8, 17, 26, 35, 44, 53, 62, 71, 80, 89, 98, 107, 116},
   {-4, 4, 13, 21, 29, 38, 46, 54, 63, 71, 79, 88, 96, 104, 113},
   {-17, -3, 10, 24, 37, 50, 64, 77, 90, 104, 117, 130, 144, 157, 170},
   {-13, -3, 7, 17, 28, 38, 48, 59, 69, 79, 90, 100, 110, 121, 131},
   {-29, -15, -2, 12, 25, 39, 52, 66, 79, 93, 106, 120, 134, 147, 161},
   {7, 16, 24, 33, 41, 49, 58, 66, 74, 83, 91, 99, 108, 116, 124},
   {4, 9, 14, 19, 25, 30, 35, 41, 46, 51, 57, 62, 67, 73, 78},
   {-1, 7, 15, 24, 32, 40, 48, 56, 64, 73, 81, 89, 97, 105, 113},
   {-4, 8, 20, 32, 44, 56, 68, 80, 91, 103, 115, 127, 139, 151, 163}},

  {{1, 11, 21, 32, 42, 53, 63, 74, 84, 95, 105, 116, 126, 137, 147},
   {8, 17, 26, 36, 45, 54, 63, 73, 82, 91, 100, 110, 119, 128, 137},
   {4, 16, 28, 40, 52, 64, 76, 88, 100, 112, 124, 136, 148, 160, 172},
   {-14, -3, 9, 21, 33, 45, 57, 68, 80, 92, 104, 116, 128, 139, 151},
   {5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 136, 146},
   {-13, 4, 22, 39, 57, 74, 92, 109, 127, 144, 161, 179, 196, 214, 231},
   {32, 44, 56, 68, 80, 93, 105, 117, 129, 141, 153, 165, 177, 189, 201},
   {14, 24, 33, 43, 53, 63, 72, 82, 92, 102, 111, 121, 131, 141, 151},
   {-26, -10, 5, 21, 36, 51, 67, 82, 97, 113, 128, 144, 159, 174, 190},
   {12, 22, 32, 42, 51, 61, 71, 81, 90, 100, 110, 120, 129, 139, 149},
   {-4, 7, 19, 31, 42, 54, 66, 77, 89, 101, 112, 124, 136, 147, 159},
   {-15, -4, 8, 19, 31, 43, 54, 66, 77, 89, 100, 112, 123, 135, 147},
   {-5, 8, 20, 33, 45, 58, 70, 83, 95, 108, 120, 133, 145, 158, 170},
   {1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 111, 122, 133, 144, 155},
   {9, 19, 28, 38, 48, 58, 67, 77, 87, 96, 106, 116, 126, 135, 145},
   {-13, -1, 11, 22, 34, 46, 57, 69, 81, 92, 104, 116, 127, 139, 151},
   {-2, 6, 15, 23, 31, 40, 48, 56, 65, 73, 82, 90, 98, 107, 115},
   {-22, -11, 0, 11, 22, 34, 45, 56, 67, 78, 89, 101, 112, 123, 134},
   {-41, -25, -9, 7, 23, 39, 55, 71, 87, 103, 119, 135, 151, 167, 183},
   {9, 19, 29, 38, 48, 58, 67, 77, 87, 97, 106, 116, 126, 135, 145}},

  {{-6, 3, 13, 22, 31, 41, 50, 59, 69, 78, 87, 97, 106, 116, 125},
   {30, 42, 55, 67, 80, 93, 105, 118, 130, 143, 156, 168, 181, 193, 206},
   {17, 30, 43, 56, 69, 83, 96, 109, 122, 135, 148, 162, 175, 188, 201},
   {20, 29, 37, 45, 54, 62, 71, 79, 88, 96, 105, 113, 122, 130, 138},
   {50, 65, 80, 94, 109, 124, 138, 153, 168, 182, 197, 211, 226, 241,
    255},
   {30, 43, 55, 68, 81, 93, 106, 119, 131, 144, 157, 169, 182, 195, 207},
   {-32, -15, 3, 20, 37, 54, 71, 89, 106, 123, 140, 157, 175, 192, 209},
   {31, 39, 47, 55, 63, 71, 80, 88, 96, 104, 112, 120, 128, 137, 145},
   {12, 21, 30, 39, 48, 57, 67, 76, 85, 94, 103, 113, 122, 131, 140},
   {0, 11, 21, 32, 43, 53, 64, 75, 85, 96, 106, 117, 128, 138, 149},
   {1, 13, 26, 38, 51, 63, 75, 88, 100, 113, 125, 138, 150, 163, 175},
   {25, 39, 52, 65, 79, 92, 106, 119, 133, 146, 160, 173, 187, 200, 213},
   {-26, -10, 6, 22, 39, 55, 71, 87, 103, 119, 136, 152, 168, 184, 200},
   {12, 22, 33, 43, 54, 65, 75, 86, 96, 107, 118, 128, 139, 149, 160},
   {11, 23, 35, 48, 60, 72, 84, 96, 108, 121, 133, 145, 157, 169, 182},
   {7, 16, 25, 34, 44, 53, 62, 71, 81, 90, 99, 108, 118, 127, 136},
   {48, 61, 74, 87, 100, 113, 125, 138, 151, 164, 177, 190, 203, 216,
    228},
   {14, 26, 37, 49, 60, 72, 83, 94, 106, 117, 129, 140, 152, 163, 175},
   {11, 22, 33, 45, 56, 67, 79, 90, 101, 113, 124, 135, 147, 158, 169},
   {-27, -16, -4, 8, 19, 31, 43, 54, 66, 78, 89, 101, 113, 124, 136}},

  {{-13, -2, 8, 19, 30, 40, 51, 62, 72, 83, 94, 104, 115, 126, 136},
   {17, 29, 40, 52, 64, 75, 87, 99, 110, 122, 134, 145, 157, 169, 180},
   {27, 35, 43, 52, 60, 69, 77, 86, 94, 102, 111, 119, 128, 136, 145},
   {-16, 0, 17, 33, 49, 66, 82, 98, 115, 131, 148, 164, 180, 197, 213},
   {13, 21, 29, 37, 45, 53, 61, 69, 78, 86, 94, 102, 110, 118, 126},
   {-19, -8, 3, 14, 26, 37, 48, 60, 71, 82, 93, 105, 116, 127, 138},
   {6, 15, 23, 31, 39, 48, 56, 64, 73, 81, 89, 98, 106, 114, 123},
   {8, 20, 32, 44, 56, 68, 80, 92, 104, 116, 128, 139, 151, 163, 175},
   {39, 54, 69, 84, 99, 114, 129, 145, 160, 175, 190, 205, 220, 235,
    250},
   {29, 43, 56, 70, 83, 97, 111, 124, 138, 151, 165, 178, 192, 206, 219},
   {5, 16, 27, 38, 49, 60, 71, 82, 93, 104, 115, 126, 137, 148, 159},
   {28, 43, 57, 72, 86, 101, 115, 130, 145, 159, 174, 188, 203, 218,
    232},
   {-19, -2, 14, 30, 46, 62, 78, 94, 110, 126, 143, 159, 175, 191, 207},
   {38, 52, 66, 80, 95, 109, 123, 137, 152, 166, 180, 194, 208, 223,
    237},
   {30, 41, 52, 62, 73, 84, 95, 105, 116, 127, 137, 148, 159, 170, 180},
   {23, 35, 48, 60, 72, 85, 97, 109, 122, 134, 146, 159, 171, 183, 196},
   {35, 45, 55, 64, 74, 84, 94, 103, 113, 123, 132, 142, 152, 161, 171},
   {-24, -14, -4, 6, 16, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116},
   {-15, -3, 10, 23, 36, 48, 61, 74, 87, 99, 112, 125, 138, 150, 163},
   {6, 16, 27, 37, 48, 58, 69, 79, 90, 100, 111, 122, 132, 143, 153}}}};