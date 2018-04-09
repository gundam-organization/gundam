#!/bin/sh
#Argument is a boolian specifying whether fine binning is to be used
cd /data/t2k/dolan/paperStuff/recoPlots
#root inputs/NeutAirV2_dpt_allStats.root inputs/rdp_dpt_p6kFHC_feb17HL2.root <<EOF
root inputs/NeutAirV2_dpt_allStats.root inputs/rdp_dpt_p6kFHC_feb17HL2.root /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/p6K_rdp_allRuns_FHC_v1.root <<EOF
double nselr_uc, nselr_md, nselr_md0p, nselr_ik, nselr_stv;
double nselt_uc, nselt_md, nselt_md0p, nselt_ik, nselt_stv;
double purit_uc, purit_md, purit_md0p, purit_ik, purit_stv;

double ntruet_uc, ntruet_md, ntruet_md0p, ntruet_ik, ntruet_stv;
double eff_uc, eff_md, eff_md0p, eff_ik, eff_stv;

double nmut, nmutpt, nmutpf, nmufpt, nmuf, nmultip, npisb, ndissb;

_file0->cd();

selectedEvents->Draw("mectopology", "weight*((cutBranch<4 && cutBranch!=0) || (cutBranch==7))" );
nselr_uc=htemp->Integral();
nselt_uc=htemp->Integral(1,3);
purit_uc=nselt_uc/nselr_uc;
trueEvents->Draw("mectopology", "weight" );
ntruet_uc=htemp->Integral(1,3);
eff_uc=nselt_uc/ntruet_uc;

selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>500) )" )
nselr_md=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>500) &&  (pMomTrue>500) )" )
nselt_md=htemp->Integral(1,3);
purit_md=nselt_md/nselr_md;
trueEvents->Draw("mectopology", "weight*(pMomTrue>500)" );
ntruet_md=htemp->Integral(1,3);
eff_md=nselt_md/ntruet_md;

selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4) || (cutBranch==7)) )" )
nselr_md0p=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4) || (cutBranch==7)) )" )
nselt_md0p=htemp->Integral(1,3);
purit_md0p=nselt_md0p/nselr_md0p;
trueEvents->Draw("mectopology", "weight" );
ntruet_md0p=htemp->Integral(1,3);
eff_md0p=nselt_md0p/ntruet_md0p;

selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>450) && (pCosThetaRec>0.4) )" )
nselr_ik=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>450) && (pCosThetaRec>0.4) && (pMomTrue>450) && (pCosThetaTrue>0.4) )" )
nselt_ik=htemp->Integral(1,3);
purit_ik=nselt_ik/nselr_ik;
trueEvents->Draw("mectopology", "weight*((pMomTrue>450) && (pCosThetaTrue>0.4))" );
ntruet_ik=htemp->Integral(1,3);
eff_ik=nselt_ik/ntruet_ik;

selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>450) && (pMomRec<1000) && (pCosThetaRec>0.4) && (muMomRec>250) && (muCosThetaRec>-0.6) )" )
nselr_stv=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>450) && (pMomRec<1000) && (pCosThetaRec>0.4) && (muMomRec>250) && (muCosThetaRec>-0.6) && (pMomTrue>450) && (pMomTrue<1000) && (pCosThetaTrue>0.4) && (muMomTrue>250) && (muCosThetaTrue>-0.6) )" )
nselt_stv=htemp->Integral(1,3);
purit_stv=nselt_stv/nselr_stv;
trueEvents->Draw("mectopology", "weight*( (pMomTrue>450) && (pMomTrue<1000) && (pCosThetaTrue>0.4) && (muMomTrue>250) && (muCosThetaTrue>-0.6) )" );
ntruet_stv=htemp->Integral(1,3);
eff_stv=nselt_stv/ntruet_stv;

_file1->cd();
selectedEvents->Draw("mectopology", "weight*((cutBranch<4 && cutBranch!=0) || (cutBranch==7))" );
nselr_uc=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>500) )" )
nselr_md=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>450) && (pCosThetaRec>0.4) )" )
nselr_ik=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*( ((cutBranch<4 && cutBranch!=0) || (cutBranch==7)) && (pMomRec>450) && (pMomRec<1000) && (pCosThetaRec>0.4) && (muMomRec>250) && (muCosThetaRec>-0.6) )" )
nselr_stv=htemp->Integral();

selectedEvents->Draw("mectopology", "weight*(cutBranch==0)" );
nmut=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*(cutBranch==1)" );
nmutpt=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*(cutBranch==2)" );
nmutpf=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*(cutBranch==3)" );
nmufpt=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*(cutBranch==4)" );
nmuf=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*(cutBranch==7)" );
nmultip=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*(cutBranch==5)" );
npisb=htemp->Integral();
selectedEvents->Draw("mectopology", "weight*(cutBranch==6)" );
ndissb=htemp->Integral();

cout << "Events in mu TPC branch: " << nmut << endl;
cout << "Events in mu TPC p TPC branch: " << nmutpt << endl;
cout << "Events in mu TPC p FGD branch: " << nmutpf << endl;
cout << "Events in mu FGD p TPC branch: " << nmufpt << endl;
cout << "Events in mu FGD branch: " << nmuf << endl;
cout << "Events in multi p branch: " << nmultip << endl;
cout << "Events in 1pisb branch: " << npisb << endl;
cout << "Events in dissb branch: " << ndissb << endl << endl;


cout << "Unconstrained purity is: " << purit_uc << endl;
cout << "Multidimenstional purity is: " << purit_md << endl;
cout << "Multidimenstional 0p purity is: " << purit_md0p << endl;
cout << "Inferred kinematics purity is: " << purit_ik << endl;
cout << "STV purity is: " << purit_stv << endl << endl;

cout << "Unconstrained efficiency is: " << eff_uc << endl;
cout << "Multidimenstional efficiency is: " << eff_md << endl;
cout << "Multidimenstional 0p efficiency is: " << eff_md0p << endl;
cout << "Inferred kinematics efficiency is: " << eff_ik << endl;
cout << "STV efficiency is: " << eff_stv << endl << endl;

cout << "Unconstrained nevts is: " << nselr_uc << endl;
cout << "Multidimenstional nevts is: " << nselr_md << endl;
cout << "Multidimenstional 0p nevts is: " << nselr_md0p << endl;
cout << "Inferred kinematics nevts is: " << nselr_ik << endl;
cout << "STV nevts is: " << nselr_stv << endl;

EOF