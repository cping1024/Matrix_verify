#include <iostream>
#include "fv_define.h"
#include "fv_workerman.h"

#include <sys/time.h>
using namespace FEATURE_VERIFYNS;

#define outMsg(FMT,ARG...)\
do{\
    char str[1024];\
    sprintf(str, "Info: "FMT, ##ARG);\
    fprintf(stderr, "%s\n", str);\
}while(0)

double getTimeOfMSeconds(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000. + tv.tv_usec/1000.;
}

struct score_id{
    score_id(float s, int64_t i)
        : score(s)
        , id(i){}

    float score;
    int64_t id;

    bool operator >(const score_id& id) const;
};
bool score_id::operator >(const score_id& id) const
{
    return score > id.score;
}
int64_t  load_libbin(std::string fnm, float* feature){
    LOG(INFO) << "Load feature from file [" << fnm << "] ...";

    /// file open
    FILE* fp = NULL;
    if((fp = fopen(fnm.c_str(), "rb")) == NULL){
        LOG(ERROR) << "Open file [" << fnm << "] failed!";
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    int64_t fsize = (int64_t)ftell(fp);
    int64_t nfeature = fsize / (FV_FEATURESZ * sizeof(float));
    if(nfeature < 1){
        LOG(ERROR) << "No feature be defined in file [" << fnm << "]";
        return 0;
    }
    if(nfeature != 1)LOG(FATAL) << "Support only one feature one file mode!";

    fseek(fp, 0, SEEK_SET);

    if(fread(feature, sizeof(float), nfeature * FV_FEATURESZ, fp) != nfeature * FV_FEATURESZ)
        LOG(FATAL) << "Load data from file [" << fnm << "] failed!";

    /*for(int i = 0; i < FV_FEATURESZ; i++){
        if(i % 10 == 0)std::cout << std::endl;
        std::cout << feature[i] << "\t";
    }
    std::cout << std::endl;*/

    fclose(fp);
    return nfeature;
}
void  init_array(float *a, int64_t N){
    assert(a != nullptr);

    for(int64_t i = 0; i < N; i++)
        a[i] = 0.1 * rand() / RAND_MAX;
}
bool diff_array(float* a, float* b, int64_t N, const float tolerance){
    assert(a != nullptr);
    assert(b != nullptr);

    double totalerr = 0.0;
    double totalval = 0.0;
    for(int64_t i = 0; i < N; i++){
        float diff = fabs(a[i] - b[i]);
        float error;

        if(a[i] != 0)
            error = diff / a[i];
        else
            error = diff;

        if(error > tolerance){
            outMsg("Data error at point (%lld)\t%f instead of %f\n", i, a[i], b[i]);
            return false;
        }
        totalerr += fabs(error);
        totalval += fabs(a[i]);
    }
    /*outMsg("Total value [%f] average [%f] error measure [%f]."
           , totalval, totalval / N, totalerr);*/
    return true;
}
void  libQueryFeatures(std::vector<int> libid, FV_WorkerMan* workerman){
    if(libid.size() == 0)return;

    int64_t has;
    int64_t free;
    for(int i = 0; i < libid.size(); i++){
        int id = libid.at(i);
        workerman->NumFeatures(id, &has);
        workerman->NumFeatureCanbeAdded(id, &free);
        std::cout << "No."
                  << id
                  << " lib has "
                  << has
                  << " features! Can add "
                  << free
                  << " features!"
                  << std::endl;
    }
}
bool  addNoLib(FV_WorkerMan* workerman, int64_t* fids, float* features, int64_t M, int libid){
    std::vector<int64_t > capability;
    std::vector<int64_t > added;
    std::vector<int64_t > curCounter;

    /// canbe added
    int64_t Tret = 0, Tcount = 0;
    Tret = workerman->NumFeatureCanbeAdded(libid, &Tcount);
    CHECK_EQ(Tret, Tcount);
    capability.push_back(Tret);

    int64_t ret, count;
    int numWorker = workerman->numWorker();
    for(int i = 0; i < numWorker; i++){
        ret = workerman->NumFeatureCanbeAdded(i, libid, &count);
        CHECK_EQ(ret, count);
        capability.push_back(ret);

        Tret -= ret;
    }
    CHECK_EQ(Tret, 0);

    /// exist
    int64_t Eret = 0, Ecount = 0;
    Eret = workerman->NumFeatures(libid, &Ecount);
    CHECK_EQ(Eret, Ecount);
    curCounter.push_back(Eret);

    for(int i = 0; i < numWorker; i++){
        ret = workerman->NumFeatures(i, libid, &count);
        CHECK_EQ(ret, count);
        curCounter.push_back(ret);

        Eret -= ret;
    }
    CHECK_EQ(Eret, 0);

    /// add feature
    if((ret = workerman->AddFeature(libid, M, fids, features)) != M){
        LOG(INFO) << "No." << libid << " lib want add " << M << " -> " << ret;
		std::cout << capability.at(0) << " " << M << std::endl;
		std::cout << capability.at(0) << " " << ret << std::endl;
        CHECK_LT(capability.at(0), M);
        CHECK_GE(capability.at(0), ret);
    }

    /// check added + canbe = pre_canbe
    Tret = workerman->NumFeatureCanbeAdded(libid, &Tcount);
    CHECK_EQ(Tret, Tcount);
    added.push_back(Tret);

    CHECK_EQ(ret + Tret, capability.at(0));

    /// check added + pre_num = cur_num
    Eret = workerman->NumFeatures(libid, &Ecount);
    CHECK_EQ(Eret, Ecount);
    CHECK_EQ(Eret, ret + curCounter.at(0));
}
bool  addLib(FV_WorkerMan* workerman, int64_t* fids, float* features, int64_t M){
    int64_t size[10] = {65535, 65536, 100000, 1, 1000000, 655360, 100, 1, 156, 1};
    int64_t count = 0;

    for(int i = 0; i < 10; i++){
        addNoLib(workerman, fids + count, features + count * FV_FEATURESZ, size[i], 0);
        addNoLib(workerman, fids + count, features + count * FV_FEATURESZ, size[i], 1);
        addNoLib(workerman, fids + count, features + count * FV_FEATURESZ, size[i], 3);
        addNoLib(workerman, fids + count, features + count * FV_FEATURESZ, size[i], 2);
        count += size[i];
    }

    addNoLib(workerman, fids + count, features + count * FV_FEATURESZ, M - count, 2);
    addNoLib(workerman, fids + count, features + count * FV_FEATURESZ, M - count, 3);
    addNoLib(workerman, fids + count, features + count * FV_FEATURESZ, M - count, 1);
    addNoLib(workerman, fids + count, features + count * FV_FEATURESZ, M - count, 0);
}
bool  cpuRawCalc(float* c_vector, float* c_matrix, float* c_output, int64_t M, int64_t N){
    float k[4];
    float b[4];

    for(int i = 0; i < 4; i++){
        k[i] = (y[i] - y[i + 1]) / (x[i] - x[i + 1]);
        b[i] = y[i] - k[i] * x[i];

        /// outMsg("No.%d k = %.15f, b = %.15f", i, k[i], b[i]);
    }
#pragma omp parallel for
    for(int64_t ifeature = 0; ifeature < M; ifeature++){
        float score = 0.0f;        
        float* libptr = c_matrix + ifeature * N;
        for(int64_t idim = 0; idim < N; idim++){
            score += libptr[idim] * c_vector[idim];
        }

        /// do k/b map
        for(int i = 1; i < 5; i++){
            if(score < x[i]){
                score = score * k[i - 1] + b[i - 1];
                break;
            }
        }
        c_output[ifeature] = ((score <= 1) ? score : 1) * 100;
        //c_output[ifeature] = score;
    }

    /// std::cout << "CPU raw calculate finished ..." << std::endl;

    /// free(trans);
    return true;
}
int test_addFeatures(){
    /// set device id
    std::vector<int> devid;
    devid.push_back(0);
    devid.push_back(2);
    devid.push_back(1);
    devid.push_back(3);

    std::vector<int64_t> gpumemsz;
    gpumemsz.push_back(10 * 1024); /// 10G
    gpumemsz.push_back(10 * 1024); /// 10G
    gpumemsz.push_back(10 * 1024); /// 5G
    gpumemsz.push_back(10 * 1024); /// 3G

    std::vector<int> libids;
    libids.push_back(0);
    /// init with use defined param
    FV_WorkerMan* workerman = new FV_WorkerMan(devid, gpumemsz);

    /// query No.0 lib number of feature can be added
    libQueryFeatures(libids, workerman);

    delete workerman;

    /// ================================================
    /// init with default param
    /// will use all gpu, each gpu mem will be 10 * 1024 MB
    workerman = new FV_WorkerMan();
    libQueryFeatures(libids, workerman);

    /// ================================================
    /// add feature method
    int64_t ret = 0;
    float* features = new float[100000 * FV_FEATURESZ];
    int64_t* fid = new int64_t[100000];

    for(int64_t i = 0; i < 100000; i++)fid[i] = i + 1;

    init_array(features, 100000 * FV_FEATURESZ);

    /// add No.0 lib
    /// add 100000 features to No.0 lib
    std::cout << "No.0 lib add 100000 features ..." << std::endl;
    workerman->AddFeature(0, 100000, fid, features);
    libQueryFeatures(libids, workerman);

    /// add 2000 features to No.0 lib
    std::cout << "No.0 lib add 2000 features ..." << std::endl;
    workerman->AddFeature(0, 2000, fid, features);
    libQueryFeatures(libids, workerman);

    /// add No.1 lib
    libids.push_back(1);
    libQueryFeatures(libids, workerman);

    std::cout << "No.1 lib add 100000 features ..." << std::endl;
    workerman->AddFeature(1, 100000, fid, features);
    libQueryFeatures(libids, workerman);

    /// add No.2 lib
    libids.push_back(2);
    libQueryFeatures(libids, workerman);

    std::cout << "No.2 lib add 100000 features ..." << std::endl;
    workerman->AddFeature(2, 100000, fid, features);
    libQueryFeatures(libids, workerman);

    /// add No.0 lib
    std::cout << "No.0 lib add 1 features ..." << std::endl;
    workerman->AddFeature(0, 1, fid, features);
    libQueryFeatures(libids, workerman);

    std::cout << "No.0 lib add 100000 features ..." << std::endl;
    workerman->AddFeature(0, 100000, fid, features);
    libQueryFeatures(libids, workerman);

    /// add No.2 lib
    std::cout << "No.2 lib add 100000 features ..." << std::endl;
    workerman->AddFeature(2, 100000, fid, features);
    libQueryFeatures(libids, workerman);

    delete []features;
    delete []fid;

    delete workerman;
    return 0;
}
int test_featureVerify(){
    /// do cpu worker

    int64_t M = 5000000;
    int topN = 100;

    float* features = new float[M * FV_FEATURESZ];
    int64_t* fids = new int64_t[M];
    float* fv_feature = new float[FV_FEATURESZ];
    float* def_feature = new float[4 * FV_FEATURESZ];

    float*   h_topScores = new float[topN];
    int64_t* h_topID = new int64_t[topN];
    float*   d_topScores = new float[topN];
    int64_t* d_topID = new int64_t[topN];

    /// load defined features!
    std::vector<int > libids;
    libids.push_back(0);
    libids.push_back(1);
    libids.push_back(2);
    libids.push_back(3);

    std::vector<int> devid;
    devid.push_back(0);
    devid.push_back(1);
    devid.push_back(2);
    devid.push_back(3);

    std::vector<int64_t> gpumemsz;
    gpumemsz.push_back(10 * 1024); /// 10G
    gpumemsz.push_back(10 * 1024); /// 5G
    gpumemsz.push_back(10 * 1024); /// 3G
    gpumemsz.push_back(10 * 1024); /// 3G

    for(int64_t i = 0; i < M; i++)fids[i] = i;
    /// init lib features
    /// ---------------------------------------------------------
    /// test for all feature verify

    init_array(features, M * FV_FEATURESZ);
    FV_WorkerMan* workerman = new FV_WorkerMan(devid, gpumemsz);
    addLib(workerman, fids, features, M);

    for(int64_t times = 0; times < M; times++){
        LOG(INFO) << "No." << times << " working ...";

        memcpy(fv_feature, features + times * FV_FEATURESZ, FV_FEATURESZ * sizeof(float));

        /// do cpu job
        float* cpu_score = new float[M];
        cpuRawCalc(fv_feature, features, cpu_score, M, FV_FEATURESZ);

        std::vector<score_id > scores;
        for(int64_t i = 0; i < M; i++)scores.push_back(score_id(cpu_score[i], fids[i]));

        std::sort(scores.begin(), scores.end(), std::greater<score_id>());
        for(int i = 0; i < topN; i++){
            h_topScores[i] = scores.at(i).score;
            h_topID[i] = scores.at(i).id;
        }
        /// std::sort(cpu_score, cpu_score + M, std::greater<float>());
        /// memcpy(h_topScores, cpu_score, sizeof(float) * topN);
        delete []cpu_score;

        /// do gpu job
        // FV_WorkerMan* workerman = new FV_WorkerMan(devid, gpumemsz);
        // addLib(workerman, fids, features, M);
        ///libQueryFeatures(libids, workerman);

        for(int i = 0; i < 4; i++){
            LOG(INFO) << "No." << times << "]-> Do feature verify No." << i << " lib ...";
            workerman->FeatureVerify(fv_feature, i, d_topScores, d_topID, topN);
            /// outMsg("Feature verify with gpu use time %f ms", endT - startT);

            /// ==============================================================
            /// compare cpu <--> gpu
            /// LOG(INFO) << "Compare cpu/gpu result ...";
            diff_array(d_topScores, h_topScores, topN, 1.0e-4);
            /*if(fabs(d_topScores[0] - 100) >= 1.0e-4)
                CHECK_EQ(d_topScores[0], 100);*/
            //CHECK_EQ(d_topID[0], times);
            //CHECK_EQ(h_topID[0], times);

            LOG(INFO) << "Top: " << d_topID[0] << ", " << d_topScores[0]
                      << "\t" << d_topID[1] << ", " << d_topScores[1];
            /// LOG(INFO) << "Compare cpu/gpu successful!";
        }
        workerman->FeatureVerify(fv_feature, d_topScores, d_topID, topN);
        for(int i = 1; i < 4; i++){
            if(fabs(d_topScores[i] - d_topScores[0]) >= 1.0e-4)
                LOG(INFO) << "Top: " << d_topScores[0] << " vs " << d_topScores[i];
        }
        CHECK_EQ(d_topID[0], d_topID[1]);
        CHECK_EQ(d_topID[0], d_topID[2]);
        CHECK_EQ(d_topID[0], d_topID[3]);


    }
    delete workerman;
    /// ---------------------------------------------------------
    /// test for each def features!
    if(0){
        for(int64_t times = 20000; times < M; times++){
            //for(int jfeatures = 0; jfeatures < 4; jfeatures++){
                LOG(INFO) << "No." << times << " working ...";
                init_array(features, M * FV_FEATURESZ);
                memcpy(features + times * FV_FEATURESZ, def_feature, FV_FEATURESZ * sizeof(float));

                memcpy(fv_feature, def_feature, FV_FEATURESZ * sizeof(float));

                /// do cpu job
                float* cpu_score = new float[M];
                cpuRawCalc(fv_feature, features, cpu_score, M, FV_FEATURESZ);

                std::vector<score_id > scores;
                for(int64_t i = 0; i < M; i++)scores.push_back(score_id(cpu_score[i], fids[i]));

                std::sort(scores.begin(), scores.end(), std::greater<score_id>());
                for(int i = 0; i < topN; i++){
                    h_topScores[i] = scores.at(i).score;
                    h_topID[i] = scores.at(i).id;
                }
                /// std::sort(cpu_score, cpu_score + M, std::greater<float>());
                /// memcpy(h_topScores, cpu_score, sizeof(float) * topN);
                delete []cpu_score;

                /// do gpu job
                FV_WorkerMan* workerman = new FV_WorkerMan(devid, gpumemsz);
                addLib(workerman, fids, features, M);
                ///libQueryFeatures(libids, workerman);

                for(int i = 0; i < 4; i++){
                    LOG(INFO) << "No." << times << "]-> Do feature verify No." << i << " lib ...";
                    workerman->FeatureVerify(fv_feature, i, d_topScores, d_topID, topN);
                    /// outMsg("Feature verify with gpu use time %f ms", endT - startT);

                    /// ==============================================================
                    /// compare cpu <--> gpu
                    /// LOG(INFO) << "Compare cpu/gpu result ...";
                    diff_array(d_topScores, h_topScores, topN, 1.0e-4);
                    if(fabs(d_topScores[0] - 100) >= 1.0e-4)
                        CHECK_EQ(d_topScores[0], 100);
                    CHECK_EQ(d_topID[0], times);
                    CHECK_EQ(h_topID[0], times);

                    LOG(INFO) << "Top: " << d_topID[0] << ", " << d_topScores[0]
                              << "\t" << d_topID[1] << ", " << d_topScores[1];
                    /// LOG(INFO) << "Compare cpu/gpu successful!";
                }
                workerman->FeatureVerify(fv_feature, d_topScores, d_topID, topN);
                for(int i = 0; i < 1; i++){
                    if(fabs(d_topScores[i] - 100) >= 1.0e-4)
                        CHECK_EQ(d_topScores[i], 100);
                }
                /*CHECK_EQ(d_topID[0], d_topID[1]);
                CHECK_EQ(d_topID[0], d_topID[2]);
                CHECK_EQ(d_topID[0], d_topID[3]);*/

                delete workerman;
            //}
        }
    }





    /*LOG(INFO) << "Init features lib ...";
    init_array(features, M * FV_FEATURESZ);
    memcpy(features + 2256324 * FV_FEATURESZ, def_feature, FV_FEATURESZ * sizeof(float));

    memcpy(fv_feature, def_feature, FV_FEATURESZ * sizeof(float));
    /// init_array(fv_feature, FV_FEATURESZ);
    /// memcpy(fv_feature, features + 2256324 * FV_FEATURESZ, FV_FEATURESZ);

    /// ==============================================================
    /// do cpu check
    LOG(INFO) << "Do feature verify with cpu ...";
    double startT = getTimeOfMSeconds();

    float* cpu_score = new float[M];
    cpuRawCalc(fv_feature, features, cpu_score, M, FV_FEATURESZ);

    std::vector<score_id > scores;
    for(int64_t i = 0; i < M; i++)scores.push_back(score_id(cpu_score[i], fids[i]));

    std::sort(scores.begin(), scores.end(), std::greater<score_id>());
    for(int i = 0; i < topN; i++){
        h_topScores[i] = scores.at(i).score;
        h_topID[i] = scores.at(i).id;
    }
    /// std::sort(cpu_score, cpu_score + M, std::greater<float>());
    /// memcpy(h_topScores, cpu_score, sizeof(float) * topN);
    delete []cpu_score;

    double endT = getTimeOfMSeconds();
    outMsg("Feature verify with cpu use time %f ms", endT - startT);

    /// ==============================================================
    /// upload feature lib 2 gpu
    std::vector<int > libids;
    libids.push_back(0);
    libids.push_back(1);
    libids.push_back(2);
    libids.push_back(3);

    std::vector<int> devid;
    devid.push_back(0);
    devid.push_back(1);
    devid.push_back(2);
    devid.push_back(3);

    std::vector<int64_t> gpumemsz;
    gpumemsz.push_back(10 * 1024); /// 10G
    gpumemsz.push_back(10 * 1024); /// 5G
    gpumemsz.push_back(10 * 1024); /// 3G
    gpumemsz.push_back(10 * 1024); /// 3G

    FV_WorkerMan* workerman = new FV_WorkerMan(devid, gpumemsz);
    LOG(INFO) << "Add feature lib to device ...";
    // workerman->AddFeature(0, M, fids, features);
    addLib(workerman, fids, features, M);
    libQueryFeatures(libids, workerman);

    /// do feature verify
    for(int i = 0; i < 4; i++){
        LOG(INFO) << "Do feature verify No." << i << " lib ...";
        startT = getTimeOfMSeconds();
        workerman->FeatureVerify(fv_feature, i, d_topScores, d_topID, topN);
        endT = getTimeOfMSeconds();
        outMsg("Feature verify with gpu use time %f ms", endT - startT);

        /// ==============================================================
        /// compare cpu <--> gpu
        LOG(INFO) << "Compare cpu/gpu result ...";
        diff_array(d_topScores, h_topScores, topN, 1.0e-4);

        LOG(INFO) << "Compare cpu/gpu successful!";
    }

    LOG(INFO) << "Feature verify with all lib!";
    startT = getTimeOfMSeconds();
    workerman->FeatureVerify(fv_feature, d_topScores, d_topID, topN);
    endT = getTimeOfMSeconds();
    outMsg("Feature verify with gpu use time %f ms", endT - startT);

    for(int i = 0; i < topN; i++){
        if(i % 10 == 0)std::cout << std::endl;
        std::cout << d_topScores[i] << "\t";
    }
    std::cout << std::endl;

    std::cout << std::endl;
    for(int i = 0; i < topN; i++){
        if(i % 10 == 0)std::cout << std::endl;
        std::cout << d_topID[i] << "\t";
    }
    std::cout << std::endl;

    delete workerman;*/

    delete []h_topScores;
    delete []h_topID;
    delete []d_topScores;
    delete []d_topID;
    delete []fids;
    delete []features;
    delete []def_feature;

    return 0;
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();

    /// do add feature test
    // return test_addFeatures();

    /// do feature verify test
    return test_featureVerify();
}
