/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "ORBextractor.h"

// 시간 측정
#include <chrono>


using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    const float OctaveDif = 2.0f;
	const float IntraOctaveDif = 1.5f;

    static bool compare_response(KeyPoint first, KeyPoint second)
    {
        if (first.response < second.response) return false;
        else return true;
    }

    ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST):
            nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
            iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        cerr << "Feature Type is BRISK" << endl;

		std::vector<float> rList;
  		std::vector<int> nList;

		// this is the standard pattern found to be suitable also
		rList.resize(5);
		nList.resize(5);
		const double f = 0.85*0.5;

		rList[0] = (float)(f * 0.);
		rList[1] = (float)(f * 2.9);
		rList[2] = (float)(f * 4.9);
		rList[3] = (float)(f * 7.4);
		rList[4] = (float)(f * 10.8);

		nList[0] = 1;
		nList[1] = 10;
		nList[2] = 14;
		nList[3] = 15;
		nList[4] = 20;

		std::vector<int> indexChange=std::vector<int>();

        feature_ORB = ORB::create(2000, 1.2f, 1, EDGE_THRESHOLD, 0, 2, ORB::HARRIS_SCORE, PATCH_SIZE, 20);
        feature_ORB_min = ORB::create(2000, 1.2f, 1, EDGE_THRESHOLD, 0, 2, ORB::HARRIS_SCORE, PATCH_SIZE, minThFAST);

		// feature_BRISK = BRISK::create(15, 4, rList, nList, 9.75f, 13.67f, indexChange);
        // feature_BRISK_min = BRISK::create(7, 4, rList, nList, 9.75f, 13.67f, indexChange);
        // feature_BRISK_mid = BRISK::create(11, 4, rList, nList, 9.75f, 13.67f, indexChange);

        feature_BRISK = BRISK::create(16, 4, rList, nList, 5.85f*0.5, 8.2f*0.5, indexChange);
        feature_BRISK_min = BRISK::create(8, 4, rList, nList, 5.85f*0.5, 8.2f*0.5, indexChange);
        feature_BRISK_mid = BRISK::create(11, 4, rList, nList, 5.85f*0.5, 8.2f*0.5, indexChange);

        // feature_BRISK = BRISK::create(15);
        // feature_BRISK_min = BRISK::create(7);

        // BRISK parameter test

        // if (nfeatures == 1){
        //     feature_BRISK = BRISK::create(15, 4, rList, nList, 5.85f, 8.2f, indexChange);
        //     feature_BRISK_min = BRISK::create(10, 4, rList, nList, 5.85f, 8.2f, indexChange);
        // }
        // else if (nfeatures == 2){
        //     feature_BRISK = BRISK::create(20, 4, rList, nList, 5.85f, 8.2f, indexChange);
        //     feature_BRISK_min = BRISK::create(10, 4, rList, nList, 5.85f, 8.2f, indexChange);
        // }
        // else if (nfeatures == 3){
        //     feature_BRISK = BRISK::create(15, 4, rList, nList, 9.75f, 13.67f, indexChange);
        //     feature_BRISK_min = BRISK::create(10, 4, rList, nList, 9.75f, 13.67f, indexChange);
        // }
        // else if (nfeatures == 4){
        //     feature_BRISK = BRISK::create(20, 4, rList, nList, 9.75f, 13.67f, indexChange);
        //     feature_BRISK_min = BRISK::create(10, 4, rList, nList, 9.75f, 13.67f, indexChange);
        // }

        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0]=1.0f;
        mvLevelSigma2[0]=1.0f;

        // BRISK의 옥타브
        // octave       : 1.0 -> 2.0 -> 4.0 -> 8.0
        // intra-octave : 1.5 -> 3.0 -> 6.0 -> 12.0
		for (int i = 1; i < nlevels; i++) {
			if (i == 1) {	// i가 1이면 IntraOctave
				mvScaleFactor[i] = mvScaleFactor[i - 1] * IntraOctaveDif;
			}
			else {	// 이후는 octave와 intra-octave 각각 1/2
				mvScaleFactor[i] = mvScaleFactor[i - 2] * OctaveDif;
			}
			mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
		}

        // ORB의 옥타브
        // for(int i=1; i<nlevels; i++)
        // {
        //     mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        //     mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
        // }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    }

    void ORBextractor::DetectKeyPointsBRISK(std::vector<cv::KeyPoint> &keypoints)
    {
        const int nCols = 3;
        const int nRows = 3;
        const int wCell = ceil(mvImagePyramid[0].cols/nCols);  // 752/3
        const int hCell = ceil(mvImagePyramid[0].rows/nRows);  // 480/3
        const int Offset = 15;

        // const int wCell = mvImagePyramid[0].cols / 3;   // 752 -> 251       640 -> 213
        // const int hCell = mvImagePyramid[0].rows / 3;   // 480 -> 160
        // const int Offset = 15;

        for (int i = 0; i < nRows; i++)
        {
            float iniY = (i * (hCell-Offset));                          // 0    -> 145  -> 290
            float maxY = (((i+1) * hCell) + ((nRows-1-i) * Offset));    // 190  -> 335  -> 480
            if (iniY >= maxY-(hCell+2*Offset))
                iniY = maxY-(hCell+2*Offset);
            if (maxY >= mvImagePyramid[0].rows)
                maxY = mvImagePyramid[0].rows;
            
            for (int j = 0; j < nCols; j++)
            {
                float iniX = (j * (wCell-Offset));
                float maxX = (((j+1) * wCell) + ((nCols-1-j) * Offset));
                if (iniX >= maxX-(wCell+2*Offset))
                    iniX = maxX-(wCell+2*Offset);
                if (maxX > mvImagePyramid[0].cols)
                    maxX = mvImagePyramid[0].cols;
                vector<cv::KeyPoint> vKeysCell;

                feature_BRISK->detect(mvImagePyramid[0].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, Mat());
                //cerr << "first keys : " << vKeysCell.size() << " ";
                if (vKeysCell.size() < 50)
                {
                    vKeysCell.clear();
                    feature_BRISK_min->detect(mvImagePyramid[0].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, Mat());
                    //cerr << "modified keys : " << vKeysCell.size() << endl;
                    //cv::KeyPointsFilter::retainBest(vKeysCell, 300);
                }
                if (vKeysCell.size() > 500)
                    cv::KeyPointsFilter::retainBest(vKeysCell, 400);
                
                if (!vKeysCell.empty())
                {
                    for (vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x+=iniX;
                        (*vit).pt.y+=iniY;
                        keypoints.push_back(*vit);
                    }
                } 
            }
        }
        //cerr << "all keys : " << keypoints.size() << endl;

        //cv::KeyPointsFilter::retainBest(keypoints, 2000);
   
    }

    void ORBextractor::DetectKeyPointsORB(std::vector<cv::KeyPoint> &keypoints)
    {
        const int nCols = 3;
        const int nRows = 3;
        const int wCell = ceil(mvImagePyramid[0].cols/nCols);  // 752/3
        const int hCell = ceil(mvImagePyramid[0].rows/nRows);  // 480/3

        for (int i = 0; i < nRows; i++)
        {
            float iniY =i*hCell;
            float maxY = iniY+hCell;

            if(iniY>=2*hCell)
                iniY = 2*hCell;
            if(maxY>mvImagePyramid[0].rows)
                maxY = mvImagePyramid[0].rows;

            for(int j=0; j<nCols; j++)
            {
                float iniX = j*wCell;
                float maxX = iniX+wCell+6;
                if(iniX>=2*wCell)
                    iniX = 2*wCell;
                if(maxX>mvImagePyramid[0].cols)
                    maxX = mvImagePyramid[0].cols;

                vector<cv::KeyPoint> vKeysCell;

                feature_ORB->detect(mvImagePyramid[0].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, Mat());
                //FAST(image.rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, iniThFAST, true);
                if (vKeysCell.empty())
                {
                    feature_ORB_min->detect(mvImagePyramid[0].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, Mat());
                    //FAST(image.rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, minThFAST, true);
                }
                
                if (!vKeysCell.empty())
                {
                    for (vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        keypoints.push_back(*vit);
                    }
                }
            }
        }
        sort(keypoints.begin(), keypoints.end(), compare_response);
        for (int i = keypoints.size(); i > 2000; i--){
            keypoints.pop_back();
        }
    }


    int ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea)
    {


        //cout << "[ORBextractor]: Max Features: " << nfeatures << endl;
        if(_image.empty())
            return -1;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );

        // Pre-compute the scale pyramid
        ComputePyramid(image);


        // 1-1
        // 이미지를 나눠서 opencv ORB로 뽑는 방법

        //DetectKeyPointsORB(_keypoints);


        // 1-2
        // 이미지를 나눠서 opencv BRISK로 뽑는 방법
        // std::chrono::system_clock::time_point start = chrono::system_clock::now();
        DetectKeyPointsBRISK(_keypoints);
        // std::chrono::system_clock::time_point end = chrono::system_clock::now();
        // chrono::nanoseconds nano = end - start;
        // cerr << "key time : " << nano.count() << endl;

        // duration = (double)(finish-start)/CLOCKS_PER_SEC;
        // cerr << "keypoint duration : " << duration << "초" << endl;

        // 1-3
        // 이미지를 나누지 않고 opencv BRISK로 뽑는 방법
        // feature_BRISK->detect(image, _keypoints, Mat());
        // if (_keypoints.size() < 1000 && _keypoints.size() > 600)
        // {
        //     cerr << "mid Keys" << endl;
        //     _keypoints.clear();
        //     feature_BRISK_mid->detect(image, _keypoints, Mat());
        // }
        // if (_keypoints.size() < 1000)
        // {
        //     cerr << "min Keys" << endl;
        //     _keypoints.clear();
        //     feature_BRISK_min->detect(image, _keypoints, Mat());
        // }

        // 1-4
        // 이미지를 나누지 않고 opencv ORB로 뽑는 방법
        // feature_ORB->detect(image, _keypoints, Mat());
        // if (_keypoints.size() < 1000){
        //     cerr << "new Keys" << endl;
        //     _keypoints.clear();
        //     feature_ORB_min->detect(image, _keypoints, Mat());
        // }

        // std::chrono::system_clock::time_point start_2 = chrono::system_clock::now();
        feature_BRISK->compute(image, _keypoints, _descriptors);
        // std::chrono::system_clock::time_point end_2 = chrono::system_clock::now();
        // chrono::nanoseconds nano_2 = end_2 - start_2;
        // cerr << "desc time : " << nano_2.count() << endl;
        //cerr << "descriptor -> " << _descriptors.cols() << endl;

		int monoIndex = 0, stereoIndex = _keypoints.size() - 1;

        for (int level = 0; level < nlevels; ++level){
			
			for (vector<KeyPoint>::iterator keypoint = _keypoints.begin(),
						keypointEnd = _keypoints.end(); keypoint != keypointEnd; ++keypoint){
				//if (level != 0){ keypoint->pt *= scale; }
				if (keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]){
					stereoIndex--;
				}
				else{ monoIndex++; }
			}
		}

        cerr << "keys : " << _keypoints.size() << endl;


        return monoIndex;
    }

    void ORBextractor::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
            Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
            Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if( level != 0 )
            {
                resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101+BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
        }  
    }

} //namespace ORB_SLAM