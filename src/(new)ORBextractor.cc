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
		const double f = 0.85*0.5f;
        
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

		// feature_BRISK = BRISK::create(20, 4, rList, nList, 9.75f*0.5f, 13.67f*0.5f, indexChange);
        // feature_BRISK_min = BRISK::create(7, 4, rList, nList, 9.75f*0.5f, 13.67f*0.5f, indexChange);
        // feature_BRISK_mid = BRISK::create(11, 4, rList, nList, 9.75f*0.5f, 13.67f*0.5f, indexChange);

        // feature_BRISK = BRISK::create(15, 4, rList, nList, 5.85f, 8.2f, indexChange);
        // feature_BRISK_min = BRISK::create(7, 4, rList, nList, 5.85f, 8.2f, indexChange);
        // feature_BRISK_mid = BRISK::create(11, 4, rList, nList, 5.85f, 8.2f, indexChange);

        // keypoint를 위한 BRISK
        feature_BRISK = BRISK::create(20, 4, rList, nList, 5.85f * 0.5f, 8.2f * 0.5f, indexChange);
        feature_BRISK_min = BRISK::create(7, 4, rList, nList, 5.85f * 0.5f, 8.2f * 0.5f, indexChange);
        feature_BRISK_mid = BRISK::create(11, 4, rList, nList, 5.85f * 0.5f, 8.2f * 0.5f, indexChange);

        // descriptor를 위한 BRISK
        descriptor_BRISK = BRISK::create(20, 4, rList, nList, 5.85f * 0.5f, 8.2f * 0.5f, indexChange);


        feature_AGAST = AgastFeatureDetector::create(15, true, AgastFeatureDetector::OAST_9_16);
        feature_AGAST_min = AgastFeatureDetector::create(7, true, AgastFeatureDetector::OAST_9_16);


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

        //This is for orientation
        // pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }
    }

    static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
    {
        int m_01 = 0, m_10 = 0;

        const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

        // Treat the center line differently, v=0
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        int step = (int)image.step1();
        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        return fastAtan2((float)m_01, (float)m_10);
    }

    void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
        const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

        //Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x+halfX,UL.y);
        n1.BL = cv::Point2i(UL.x,UL.y+halfY);
        n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x,UL.y+halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x,BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        //Associate points to childs
        for(size_t i=0;i<vKeys.size();i++)
        {
            const cv::KeyPoint &kp = vKeys[i];
            if(kp.pt.x<n1.UR.x)
            {
                if(kp.pt.y<n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if(kp.pt.y<n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        if(n1.vKeys.size()==1)
            n1.bNoMore = true;
        if(n2.vKeys.size()==1)
            n2.bNoMore = true;
        if(n3.vKeys.size()==1)
            n3.bNoMore = true;
        if(n4.vKeys.size()==1)
            n4.bNoMore = true;
    }

    static bool compareNodes(pair<int,ExtractorNode*>& e1, pair<int,ExtractorNode*>& e2){
        if(e1.first < e2.first){
            return true;
        }
        else if(e1.first > e2.first){
            return false;
        }
        else{
            if(e1.second->UL.x < e2.second->UL.x){
                return true;
            }
            else{
                return false;
            }
        }
    }

    vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                         const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
    {
        // Compute how many initial nodes
        const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

        const float hX = static_cast<float>(maxX-minX)/nIni;

        list<ExtractorNode> lNodes;

        vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);

        for(int i=0; i<nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
            ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
            ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
            ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        //Associate points to childs
        for(size_t i=0;i<vToDistributeKeys.size();i++)
        {
            const cv::KeyPoint &kp = vToDistributeKeys[i];
            vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
        }

        list<ExtractorNode>::iterator lit = lNodes.begin();

        while(lit!=lNodes.end())
        {
            if(lit->vKeys.size()==1)
            {
                lit->bNoMore=true;
                lit++;
            }
            else if(lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }


        bool bFinish = false;

        int iteration = 0;

        vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size()*4);

        while(!bFinish)
        {
            iteration++;

            int prevSize = lNodes.size();

            lit = lNodes.begin();

            int nToExpand = 0;

            vSizeAndPointerToNode.clear();

            while(lit!=lNodes.end())
            {
                if(lit->bNoMore)
                {
                    // If node only contains one point do not subdivide and continue
                    lit++;
                    continue;
                }
                else
                {
                    // If more than one point, subdivide
                    ExtractorNode n1,n2,n3,n4;
                    lit->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit=lNodes.erase(lit);
                    continue;
                }
            }

            // Finish if there are more nodes than required features
            // or all nodes contain just one point
            if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
            {
                bFinish = true;
            }
            else if(((int)lNodes.size()+nToExpand*3)>N)
            {

                while(!bFinish)
                {

                    prevSize = lNodes.size();

                    vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end(),compareNodes);
                    for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                    {
                        ExtractorNode n1,n2,n3,n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                        // Add childs if they contain points
                        if(n1.vKeys.size()>0)
                        {
                            lNodes.push_front(n1);
                            if(n1.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n2.vKeys.size()>0)
                        {
                            lNodes.push_front(n2);
                            if(n2.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n3.vKeys.size()>0)
                        {
                            lNodes.push_front(n3);
                            if(n3.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n4.vKeys.size()>0)
                        {
                            lNodes.push_front(n4);
                            if(n4.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                        if((int)lNodes.size()>=N)
                            break;
                    }

                    if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                        bFinish = true;

                }
            }
        }


        // Retain the best point in each node
        vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(nfeatures);
        for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
        {
            vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
            cv::KeyPoint* pKP = &vNodeKeys[0];
            float maxResponse = pKP->response;

            for(size_t k=1;k<vNodeKeys.size();k++)
            {
                if(vNodeKeys[k].response>maxResponse)
                {
                    pKP = &vNodeKeys[k];
                    maxResponse = vNodeKeys[k].response;
                }
            }

            vResultKeys.push_back(*pKP);
        }

        return vResultKeys;
    }

    static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
    {
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                     keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
        {
            keypoint->angle = IC_Angle(image, keypoint->pt, umax);
        }
    }


    // ORB-SLAM3의 특징점 추출 방법과 거의 같게 진행한 방법

    void ORBextractor::DetectKeyPointsbyFAST(vector<KeyPoint> &_keypoints)
    {
        vector<vector<KeyPoint>> allKeypoints;
        allKeypoints.resize(nlevels);

        const float W = 35;

        for (int level = 0; level < nlevels; ++level)
        {
            const int minBorderX = EDGE_THRESHOLD-3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
            const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

            vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures*10);

            const float width = (maxBorderX-minBorderX);
            const float height = (maxBorderY-minBorderY);

            const int nCols = width/W;
            const int nRows = height/W;
            const int wCell = ceil(width/nCols);
            const int hCell = ceil(height/nRows);

            for(int i=0; i<nRows; i++)
            {
                const float iniY =minBorderY+i*hCell;
                float maxY = iniY+hCell+6;

                if(iniY>=maxBorderY-3)
                    continue;
                if(maxY>maxBorderY)
                    maxY = maxBorderY;

                for(int j=0; j<nCols; j++)
                {
                    const float iniX =minBorderX+j*wCell;
                    float maxX = iniX+wCell+6;
                    if(iniX>=maxBorderX-6)
                        continue;
                    if(maxX>maxBorderX)
                        maxX = maxBorderX;

                    vector<cv::KeyPoint> vKeysCell;

                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell,iniThFAST,true);
                    //feature_AGAST->detect(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, Mat());
                    //AGAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell,iniThFAST,true, AgastFeatureDetector::OAST_9_16);
                    //feature_BRISK->detect(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, Mat());

                    if(vKeysCell.empty())
                    {
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell,minThFAST,true);
                        //feature_AGAST_min->detect(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, Mat());
                        //AGAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell,minThFAST,true, AgastFeatureDetector::OAST_9_16);
                        //feature_BRISK_min->detect(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, Mat());
                    }

                    if(!vKeysCell.empty())
                    {
                        for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                        {
                            (*vit).pt.x+=j*wCell;
                            (*vit).pt.y+=i*hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }
                }
            }
            
            vector<KeyPoint> & keypoints = allKeypoints[level];
            keypoints.reserve(nfeatures);

            keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                          minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

            const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

            // Add border to coordinates and scale information
            const int nkps = keypoints.size();
            for(int i=0; i<nkps ; i++)
            {
                keypoints[i].pt.x+=minBorderX;
                keypoints[i].pt.y+=minBorderY;
                keypoints[i].octave=level;
                keypoints[i].size = scaledPatchSize;
            }
        }

        for (int level = 0; level < nlevels; level++)
        {
            const float scale = mvScaleFactor[level];
            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
            const int keys = allKeypoints[level].size();
            for (int i = 0; i < keys; i++)
                allKeypoints[level][i].pt *= scale;
            _keypoints.insert(_keypoints.end(), allKeypoints[level].begin(), allKeypoints[level].end());
        }
    }

    // FAST 특징점 + BRISK 기술자
    void ORBextractor::DetectKeyPointsBRISK(std::vector<cv::KeyPoint> &keypoints)
    {
        const float W = 35;

        for (int level = 0; level < nlevels; ++level)
        {
            const int minBorderX = EDGE_THRESHOLD-3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
            const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

            vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures*10);

            const float width = (maxBorderX-minBorderX);
            const float height = (maxBorderY-minBorderY);

            const int nCols = width/W;
            const int nRows = height/W;
            const int wCell = ceil(width/nCols);
            const int hCell = ceil(height/nRows);

            for(int i=0; i<nRows; i++)
            {
                const float iniY =minBorderY+i*hCell;
                float maxY = iniY+hCell+6;

                if(iniY>=maxBorderY-3)
                    continue;
                if(maxY>maxBorderY)
                    maxY = maxBorderY;

                for(int j=0; j<nCols; j++)
                {
                    const float iniX =minBorderX+j*wCell;
                    float maxX = iniX+wCell+6;
                    if(iniX>=maxBorderX-6)
                        continue;
                    if(maxX>maxBorderX)
                        maxX = maxBorderX;

                    vector<cv::KeyPoint> vKeysCell;

                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                          vKeysCell,iniThFAST,true);
                    
                    if(vKeysCell.empty())
                    {
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,minThFAST,true);
                    }

                    if(!vKeysCell.empty())
                    {
                        for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                        {
                            (*vit).pt.x+=j*wCell;
                            (*vit).pt.y+=i*hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }

                }
            }
            
            cv::KeyPointsFilter::retainBest(vToDistributeKeys, ceil(mnFeaturesPerLevel[level]*2));
            computeOrientation(mvImagePyramid[level], vToDistributeKeys, umax);

            const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

            // Add border to coordinates and scale information
            const int nkps = vToDistributeKeys.size();
            const float scale = mvScaleFactor[level];
            for(int i=0; i<nkps ; i++)
            {
                vToDistributeKeys[i].pt.x+=minBorderX;
                vToDistributeKeys[i].pt.y+=minBorderY;
                vToDistributeKeys[i].octave=level;
                vToDistributeKeys[i].size = scaledPatchSize;
                vToDistributeKeys[i].pt *= scale;
            }

            // vToDistributeKeys에 담긴 keys를 keypoints로 옮기기
            keypoints.insert(keypoints.end(), vToDistributeKeys.begin(), vToDistributeKeys.end());
        }
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


        // 이미지를 나눠서 FAST로 추출하는 방법
        // std::chrono::system_clock::time_point start = chrono::system_clock::now();
        //DetectKeyPointsBRISK(_keypoints);
        DetectKeyPointsbyFAST(_keypoints);
        // std::chrono::system_clock::time_point end = chrono::system_clock::now();
        // chrono::nanoseconds nano = end - start;
        // cerr << "key time : " << nano.count() << endl;

        // duration = (double)(finish-start)/CLOCKS_PER_SEC;
        // cerr << "keypoint duration : " << duration << "초" << endl;



        // std::chrono::system_clock::time_point start_2 = chrono::system_clock::now();
        descriptor_BRISK->compute(image, _keypoints, _descriptors);
        //feature_ORB->compute(image, _keypoints, _descriptors);

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

        //cerr << "keys : " << _keypoints.size() << endl;


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