// FaceFeatureVerify.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <windows.h>
#include "ST_FaceAPI.h"
#pragma comment(lib, "STFace_API.lib")

#include <opencv2/opencv.hpp>
#pragma comment(lib, "opencv_core246.lib")
#pragma comment(lib, "opencv_highgui246.lib")
#pragma comment(lib, "opencv_imgproc246.lib")
using namespace std;
using namespace cv;

//#define STFACE_VER_472
#define STFACE_VER_520

#ifdef STFACE_VER_472
#define FEAT_HEAD_SIZE	4			/*	����ͷ��С,���ֽ�			*/
#endif // STFACE_VER_472

#ifdef STFACE_VER_520
#define FEAT_HEAD_SIZE	12			/*	����ͷ��С,���ֽ�			*/
#endif // STFACE_VER_520

#define FEAT_RAW_DIM	180			/*	������ά��				*/


//typedef unsigned char BYTE;

int FaceFeatureVerify(void* pFaceFeat1, void* pFaceFeat2, float *score);

int _tmain(int argc, _TCHAR* argv[])
{

	int nRet = 0;
	FILE *fp = NULL;
	HANDLE hFile = NULL;
	DWORD dwHasWrite;
	DWORD dwHasRead;
	int	nMaxFace	= 200;						/*	������ߴ�			*/	
	int nMinFace	= 40;						/*	��С���ߴ�			*/	
	int nFaceNum	= 10;						/*	������������		*/	

	int nFeatNum	= 0;						/*  ����������Ŀ			*/	
	int nFeatSize	= 0;						/*  ����������С			*/
	BYTE* pFaceFeat[10] = {NULL};				/*  ������������			*/
	
	float score1, score2;						/*  ���������ȶԷ���		*/

	OutputDebugStringA("start!\n");
	printf("start!\n");

	VI_Img					face_image;			/*	����ͼ��				*/
	ST_FACEAPI_HANDLE		face_handle;		/*	�㷨ͨ��handle		*/
	ST_SDKParamIn			basic_param;		/*	�㷨ͨ����������		*/
	ST_SDKParamInAdv		adv_param;			/*	�㷨ͨ���߼�����		*/
	FaceDetectResult		face_result[10];	/*	�������������		*/

	/**
	 *	Description: SDK��ʼ��
	 */
	nRet = ST_face_identify_init("192.168.12.204", 5678);
	if (nRet)
	{
		printf("[ST_face_identify_init] error code : %d��\n", nRet);
		return -1;
	}

	/**
	 *	Description: ��ȡ����Size
	 */
	nFeatSize = ST_get_face_feature_size();
	if (nFeatSize <= 0)
	{
		printf("[ST_get_face_feature_size] error code : %d��\n", nFeatSize);
		return -1;
	}


	/*	��������							*/
	basic_param.nImageWidth		 = 1920;
	basic_param.nImageHeight	 = 1080;
	basic_param.nMaxFaceSize	 = nMaxFace;
	basic_param.nMinFaceSize	 = nMinFace;

	/*	�߼�����							*/
	adv_param.procmode	= ST_FA_INIT_MODE_FL + ST_FA_INIT_MODE_FE + ST_FA_INIT_MODE_FM;
										/*	�������+������ȡ+�����ȶ�		*/
	adv_param.ncpu	= 1;				/*	CPU����							*/
	adv_param.nmode = 0;				/*	�ϸ��������						*/
	adv_param.skip_below_thresh	= TRUE;			
	adv_param.ndetect_points = 21;
	
	/**
	 *	Description: �����㷨ͨ��
	 */
	face_handle = ST_face_identify_creathandle(&basic_param, &adv_param);
	if (face_handle <= 0)
	{
		printf("[ST_face_identify_creathandle] error code : %d��\n", face_handle);
		return -1;
	}


	char szImagePath[_MAX_PATH];
	char szDatPath[_MAX_PATH];
	for (int i = 0; i < 6; ++i)
	{

		//getchar();
		sprintf_s(szImagePath, "C:\\TEST\\%d.jpg", i + 1);
		Mat test_image = imread(szImagePath);
		if (!test_image.data){
			fprintf(stderr, "fail to read %s!\n", szImagePath);
			continue;
		}

		face_image.nWidth		= test_image.cols;
		face_image.nHeight		= test_image.rows;
		face_image.nStep[0]		= test_image.step;
		face_image.iType		= ST_BGRTRIPLE;
		face_image.pData[0]		= test_image.data;

		
		/**
		 *	Description: �������
		 */
		nRet = ST_face_detection(face_handle
			, &face_image
			, nMinFace
			, nMaxFace
			, face_result
			, &nFaceNum
			, NULL);
		if (nRet)
		{
			printf("[ST_face_detection] error code : %d!\n", nRet);
			continue;
		}

		///**
		// *	Description: ��ʾ��⵽������λ�ÿ�
		// */
		//for (int i = 0; i < nFaceNum; ++i)
		//{
		//	cvRectangle(test_image, cvPoint(face_result[i].left, face_result[0].top), cvPoint(face_result[i].right, face_result[0].bottom), CV_RGB(255, 0, 0), 3);
		//}
		//imshow("result", test_image);
		//cvWaitKey(0);

		for (int j = 0; j < nFaceNum; ++j)
		{
			printf("Img[%d], Face[%d], POS:(%d %d) (%d %d)\n", i + 1, i + 1, face_result[j].left, face_result[j].top, face_result[j].right, face_result[j].bottom);
		}

		if (nFaceNum > 0)
		{
			RECT rcFacePos;
			//void *pFeatPtr = NULL;
			if (NULL == pFaceFeat[nFeatNum])
			{
				pFaceFeat[nFeatNum] = (BYTE *)malloc(sizeof(BYTE)*nFeatSize);
				if (NULL == pFaceFeat[nFeatNum])
				{
					printf("malloc feature memory failed!\n");
					continue;
				}
			}
			
			/**
			 *	Description: ��ȡ��������
			 */
			rcFacePos.left		= face_result[0].left;
			rcFacePos.right		= face_result[0].right;
			rcFacePos.top		= face_result[0].top;
			rcFacePos.bottom	= face_result[0].bottom;
			nRet = ST_feat_extract(face_handle
				, &face_image
				, &rcFacePos
				, pFaceFeat[nFeatNum]
				);
			if (nRet)
			{
				printf("[ST_feat_extract] error code : %d!\n", nRet);
				getchar();
				continue;
			}

			/**
			 *	Description: ���������ļ�
			 */
			sprintf_s(szDatPath, "C:\\TEST\\%d.dat", nFeatNum + 1);
			hFile = CreateFile(szDatPath, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
			if (hFile == INVALID_HANDLE_VALUE)
			{
				printf("can not open file : %s!\n", szDatPath);
				continue;
			}
			WriteFile(hFile, pFaceFeat[i], sizeof(BYTE)*nFeatSize, &dwHasWrite, NULL);
			CloseHandle(hFile);
			hFile = NULL;

			nFeatNum++;		
		} //if (nFaceNum > 0)


	} //for (int i = 0; i < 6; ++i)

	for (int i = 1; i < nFeatNum; ++i)
	{
		nRet = ST_verify_two_feats(face_handle, pFaceFeat[0], pFaceFeat[i], &score1);
		if (nRet)
		{
			printf("[ST_verify_two_feats] error code : %d!\n", nRet);
			continue;
		}

		nRet = FaceFeatureVerify(pFaceFeat[0], pFaceFeat[i], &score2);
		printf("1.jpg : %d.jpg score : %f, %f\n", i+1, score1*100, score2);


		if (pFaceFeat[i])
		{
			free(pFaceFeat[i]);
			pFaceFeat[i] = NULL;
		}

	} //for (int i = 1; i < nFeatNum; ++i)

	if (pFaceFeat[0])
	{
		free(pFaceFeat[0]);
		pFaceFeat[0] = NULL;
	}
	
	/**
	 *	Description: �����㷨ͨ��
	 */
	nRet = ST_face_identify_deletehandle(&face_handle);
	if (nRet)
	{
		printf("[ST_face_identify_deletehandle] error code : %d��\n", nRet);
		return -1;
	}

	/**
	 *	Description: SDK����ʼ��
	 */
	nRet = ST_face_identify_uninit();
	if (nRet)
	{
		printf("[ST_face_identify_uninit] error code : %d��\n", nRet);
		return -1;
	}


	printf("\n\n");

	for (int i = 0; i < nFeatNum; ++i)
	{
		if (NULL == pFaceFeat[i])
		{
			pFaceFeat[i] = (BYTE *)malloc(sizeof(BYTE)*nFeatSize);
			if (NULL == pFaceFeat[i])
			{
				printf("malloc feature memory failed!\n");
				continue;
			}
		}

		/**
		 *	Description: ��ȡ�����ļ�
		 */
		sprintf_s(szDatPath, "C:\\TEST\\%d.dat", i + 1);
		memset(pFaceFeat[i], 0, sizeof(BYTE)*nFeatSize);
		hFile = CreateFile(szDatPath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
		if (hFile == INVALID_HANDLE_VALUE)
		{
			printf("can not open file : %s!\n", szDatPath);
			continue;
		}
		ReadFile(hFile, pFaceFeat[i], sizeof(BYTE)*nFeatSize, &dwHasRead, NULL);
		CloseHandle(hFile);
		hFile = NULL;

		if (i > 0)
		{
			nRet = FaceFeatureVerify(pFaceFeat[0], pFaceFeat[i], &score2);
			printf("1.jpg : %d.jpg score : %f\n", i + 1, score2);
		}
	}


	for (int i = 0; i < nFeatNum; ++i)
	{
		if (pFaceFeat[i])
		{
			free(pFaceFeat[i]);
			pFaceFeat[i] = NULL;
		}
	}

	getchar();
	
	return 0;

}

int FaceFeatureVerify(void* pFaceFeat1, void* pFaceFeat2, float *score)
{
	if (NULL == pFaceFeat1
		|| NULL == pFaceFeat2
		|| NULL == score)
	{
		return -1;
	}

	/**
	 *	Description: �����ȶԷ���
	 */
	float ftScore = .0f;

	/**
	 *	Description: ������ָ��
	 */
	float *pRawFeat1 = NULL;
	float *pRawFeat2 = NULL;

	/**
	 *	Description: �ȶԷ���ӳ������
	 */
	float k[4];			/*	б��			*/
	float b[4];			/*	�ؾ�			*/


	/**
	 *	Description: ӳ�����߹յ�
	 */
	/* src score  -1  0.39  0.44  0.5  1    */
	/* dst score   0  0.5   0.7   0.9  1    */
	
	/**
	*	Description: SenseTime Face SDK v4.7.2
	*/
#ifdef STFACE_VER_472
	float x[5] = { -1, 0.39, 0.44, 0.5, 1 };
	float y[5] = { 0, 0.5, 0.7, 0.9, 1 };
#endif //STFACE_VER_472
	
	/**
	*	Description: SenseTime Face SDK v5.2.0
	*/
#ifdef STFACE_VER_520
	float x[5] = { -1, 0.43, 0.5, 0.55, 1 };
	float y[5] = { 0, 0.5, 0.7, 0.9, 1 };
#endif //STFACE_VER_520

	/**
	 *	Description: �����ߵ�б�ʺͽؾ�
	 *  б�ʣ�k = (y1-y2) / (x1-x2)
	 *  �ؾࣺb = y1 - kx1	
	 */
	for (int i = 0; i < 4; i++)
	{
		k[i] = (y[i] - y[i + 1]) / (x[i] - x[i + 1]);
		b[i] = y[i] - k[i] * x[i];
	}


	/**
	*	Description: ����������������ͷ
	*/
	pRawFeat1 = (float *)((BYTE *)pFaceFeat1 + FEAT_HEAD_SIZE);
	pRawFeat2 = (float *)((BYTE *)pFaceFeat2 + FEAT_HEAD_SIZE);
	for (int i = 0; i < FEAT_RAW_DIM; ++i)
	{
		ftScore += (pRawFeat1[i] * pRawFeat2[i]);
	}

	for (int i = 1; i < 5; i++)
	{
		if (ftScore <= x[i])
		{
			ftScore = ftScore*k[i - 1] + b[i - 1];
			break;
		}
	}

	ftScore = ftScore <= 1 ? ftScore : 1;
	*score = ftScore * 100;
	return 0;
}