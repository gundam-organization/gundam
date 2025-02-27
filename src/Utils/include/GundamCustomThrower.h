//
// Created by Lorenzo Giannessi on 30/01/2025.
//

#ifndef GUNDAM_CUSTOMTHROWER_H
#define GUNDAM_CUSTOMTHROWER_H


#include "GenericToolbox.String.h"
#include "GenericToolbox.Root.h"
#include "CmdLineParser.h"
#include "Logger.h"

#include "TDirectory.h"
#include "TObject.h"

#include <map>
#include <string>
#include <vector>
#include <utility>
#include <functional>

// compiler flags
#if HAS_CPP_17
#define GUNDAM_LIKELY_COMPILER_FLAG [[likely]]
#define GUNDAM_UNLIKELY_COMPILER_FLAG [[unlikely]]
#else
#define GUNDAM_LIKELY_COMPILER_FLAG
#define GUNDAM_UNLIKELY_COMPILER_FLAG
#endif

// dev tools
#define DEBUG_VAR(myVar) LogDebug << "DEBUG_VAR: " << GET_VAR_NAME_VALUE(myVar) << std::endl




namespace CustomThrower {


  inline void throwCorrelatedParameters(TMatrixD* choleskyCovMatrix_, std::vector<double>& thrownParListOut_){
      if( choleskyCovMatrix_ == nullptr ) return;
      if( thrownParListOut_.size() != choleskyCovMatrix_->GetNcols() ){
          thrownParListOut_.resize(choleskyCovMatrix_->GetNcols(), 0);
      }
      TVectorD thrownParVec(choleskyCovMatrix_->GetNcols());
      for( int iPar = 0 ; iPar < choleskyCovMatrix_->GetNcols() ; iPar++ ){
          thrownParVec[iPar] = gRandom->Gaus();
      }
      thrownParVec *= (*choleskyCovMatrix_);
      for( int iPar = 0 ; iPar < choleskyCovMatrix_->GetNcols() ; iPar++ ){
          thrownParListOut_.at(iPar) = thrownParVec[iPar];
      }
  }// end of function throwCorrelatedParameters(TMatrixD* choleskyCovMatrix_, std::vector<double>& thrownParListOut_)

  inline std::vector<double> throwCorrelatedParameters(TMatrixD* choleskyCovMatrix_){
      std::vector<double> out;
      throwCorrelatedParameters(choleskyCovMatrix_, out);
      return out;
  }// end of function throwCorrelatedParameters(TMatrixD* choleskyCovMatrix_)

  inline void throwCorrelatedParameters(  TMatrixD* choleskyCovMatrix_, std::vector<double>& thrownParListOut_, std::vector<double>& weights,
                                          double pedestalEntity, double pedestalLeftEdge, double pedestalRightEdge
  ){

      double pi = TMath::Pi();
      double NormalizingFactor = 1.0 / (TMath::Sqrt(2.0 * pi));
      double pedestalRange = pedestalRightEdge - pedestalLeftEdge;
      if( choleskyCovMatrix_ == nullptr ) return;
      if( thrownParListOut_.size() != choleskyCovMatrix_->GetNcols() ){
          thrownParListOut_.resize(choleskyCovMatrix_->GetNcols(), 0);
      }
      weights.resize(choleskyCovMatrix_->GetNcols(), 0);
      for( int iPar = 0 ; iPar < choleskyCovMatrix_->GetNcols() ; iPar++ ){
          weights.at(iPar) = 0;
      }

      TVectorD thrownParVec(choleskyCovMatrix_->GetNcols());
      double choice = gRandom->Uniform(0,1);
      if (choice>pedestalEntity) {
          for (int iPar = 0; iPar < choleskyCovMatrix_->GetNcols(); iPar++) {
              thrownParVec[iPar] = gRandom->Gaus();
              if (thrownParVec[iPar]>pedestalLeftEdge and thrownParVec[iPar]<pedestalRightEdge){
                  weights.at(iPar) = -TMath::Log(
                          pedestalEntity*1.0/pedestalRange + (1.0-pedestalEntity) * NormalizingFactor * TMath::Exp(-0.500 * thrownParVec[iPar] * thrownParVec[iPar])
                  );
              }else{
                  weights.at(iPar) = -TMath::Log((1.0-pedestalEntity) * NormalizingFactor )
                                          + 0.500 * thrownParVec[iPar] * thrownParVec[iPar];
              }
          }
      }else{
          for (int iPar = 0; iPar < choleskyCovMatrix_->GetNcols(); iPar++) {
              thrownParVec[iPar] = gRandom->Uniform(pedestalLeftEdge, pedestalRightEdge);
              weights.at(iPar) = -TMath::Log(
                      pedestalEntity*1.0/pedestalRange + (1.0-pedestalEntity) * NormalizingFactor * TMath::Exp(-0.500 * thrownParVec[iPar] * thrownParVec[iPar])
              );
          }
      }
      thrownParVec *= (*choleskyCovMatrix_);
      for( int iPar = 0 ; iPar < choleskyCovMatrix_->GetNcols() ; iPar++ ){
          thrownParListOut_.at(iPar) = thrownParVec[iPar];
//          LogInfo<<"{GundamUtils} thrownParVec["<<iPar<<"] = "<<thrownParVec[iPar]<<std::endl;
      }
  }// end of function throwCorrelatedParameters(TMatrixD* choleskyCovMatrix_, std::vector<double>& thrownParListOut_, std::vector<double>& weights, double pedestalEntity, double pedestalLeftEdge, double pedestalRightEdge)

  inline void throwCorrelatedParameters(TMatrixD* choleskyCovMatrix_, std::vector<double>& thrownParListOut_, std::vector<double>& weights){
      throwCorrelatedParameters(choleskyCovMatrix_, thrownParListOut_, weights, 0, 0, 0);
  }// end of function throwCorrelatedParameters(TMatrixD* choleskyCovMatrix_, std::vector<double>& thrownParListOut_, std::vector<double>& weights)

  inline void throwTStudentParameters(TMatrixD* choleskyCovMatrix_, double nu_, std::vector<double>& thrownParListOut_, std::vector<double>& weights){
      // Simple sanity check
      if( choleskyCovMatrix_ == nullptr ) return;
      if( thrownParListOut_.size() != choleskyCovMatrix_->GetNcols() ){
          thrownParListOut_.resize(choleskyCovMatrix_->GetNcols(), 0);
      }
      if( weights.size() != choleskyCovMatrix_->GetNcols() ){
          weights.resize(choleskyCovMatrix_->GetNcols(), 0);
      }
      // Throw N independent normal distributions
      TVectorD thrownParVec(choleskyCovMatrix_->GetNcols());
      for( int iPar = 0 ; iPar < choleskyCovMatrix_->GetNcols() ; iPar++ ){
          thrownParVec[iPar] = gRandom->Gaus();
      }
      // Multiply by cov matrix to obtain the gaussian part of the t-student (expanded as the covariance matrix says)
      TVectorD thrownParVecExpanded = (*choleskyCovMatrix_)*thrownParVec;
      std::vector<double> chiSquareForStudentT(choleskyCovMatrix_->GetNcols());
      int p = 1; // because only THEN I sum over all dimensions. At this level it's all independent single-dim variables
      for( int iPar = 0 ; iPar < choleskyCovMatrix_->GetNcols() ; iPar++ ){
          // Throw a chisquare with nu_ degrees of freedom
          double chiSquareProb = gRandom->Uniform(0,1);
          chiSquareForStudentT.at(iPar) = TMath::ChisquareQuantile(chiSquareProb, nu_);
          // Build the t-student throw by multiplying the multivariate gaussian (with input cov. matrix) and the chisquare
          thrownParListOut_.at(iPar) = thrownParVecExpanded[iPar] * sqrt(nu_/chiSquareForStudentT.at(iPar));
          // Fill the weights vector now (according to the pdf of the t-student multivariate distribution)
          double logFactor1 = TMath::LnGamma(0.5*(nu_+p)) - TMath::LnGamma(0.5*nu_);
          double logDenominator =  + (0.5*p)*TMath::Log(nu_*TMath::Pi());
          double normalizedTStudentThrow = thrownParVec[iPar] * sqrt(nu_/chiSquareForStudentT.at(iPar));
          double logFactor2 = -0.5*(nu_+p)*TMath::Log( 1 + 1/nu_*normalizedTStudentThrow*normalizedTStudentThrow );
          weights.at(iPar) = -(logFactor1) +(logDenominator) - logFactor2;
          //std::cout<<p<<" "<<nu_<<"  "<<-(logFactor1)<<"  "<<+logDenominator<<"  "<<-logFactor2<<std::endl;
      }

  }// end of function throwTStudentParameters(TMatrixD* choleskyCovMatrix_, double nu_, std::vector<double>& thrownParListOut_, std::vector<double>& weights)

}

#endif //GUNDAM_CUSTOMTHROWER_H
