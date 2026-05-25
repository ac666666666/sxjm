"""一键跑全流程: Q1 -> Q2 -> Q3"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import q1_yield_model
import q2_layout_mip
import q3_risk_decision

def main():
    t0 = time.time()
    print('\n========== Q1: Yield prediction ==========')
    q1_yield_model.main()
    print(f'  Q1 done in {time.time()-t0:.1f}s')

    t1 = time.time()
    print('\n========== Q2: Deterministic MIP ==========')
    q2_layout_mip.main()
    print(f'  Q2 done in {time.time()-t1:.1f}s')

    t2 = time.time()
    print('\n========== Q3: Risk-aware decision ==========')
    q3_risk_decision.main()
    print(f'  Q3 done in {time.time()-t2:.1f}s')

    print(f'\nTOTAL time: {time.time()-t0:.1f}s')
    print('Check outputs/ and figures/ for results.')

if __name__ == '__main__':
    main()
