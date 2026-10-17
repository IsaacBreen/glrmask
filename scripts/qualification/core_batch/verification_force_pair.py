"""Independent strict-forcing release comparison. No library/build edits."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,statistics,subprocess,sys
ap=argparse.ArgumentParser();ap.add_argument('--baseline-site',type=Path,required=True);ap.add_argument('--candidate-site',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repetitions',type=int,default=2000);ap.add_argument('--pairs',type=int,default=3)
a=ap.parse_args();assert a.repetitions>=20 and a.pairs>=1
assert not a.output.exists(), a.output;a.output.mkdir(parents=True)
h=Path(__file__).parent;driver=h/'review_force_bench.py'
sites={'baseline':a.baseline_site.resolve(),'candidate':a.candidate_site.resolve()}
identity={'profile':'canonical release Python, equal 8192-ID vocab, strict ID forcing only','created_at':datetime.datetime.now().astimezone().isoformat(),'pairs':a.pairs,'repetitions_per_row':a.repetitions,'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),'sites':{}}
for k,site in sites.items():
    bins=list((site/'glrmask').glob('*.pyd'));assert len(bins)==1,(site,bins)
    identity['sites'][k]={'path':str(site),'library':str(bins[0]),'sha256':hashlib.sha256(bins[0].read_bytes()).hexdigest()}
(a.output/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf8')
trials={'baseline':[],'candidate':[]};paired=[]
for i in range(a.pairs):
    current={}
    for side in (['baseline','candidate'] if i%2==0 else ['candidate','baseline']):
        output=a.output/f'pair-{i+1}-{side}.json';log=a.output/f'pair-{i+1}-{side}.log'
        print('RUN',i+1,side,flush=True)
        with log.open('wb') as f:
            r=subprocess.run([sys.executable,str(driver),'--site',str(sites[side]),'--output',str(output),'--variant',side,'--vocab-size','8192','--repetitions',str(a.repetitions)],stdout=f,stderr=subprocess.STDOUT,timeout=600)
        assert r.returncode==0,(side,r.returncode,log)
        d=json.loads(output.read_text(encoding='utf8'));indexed={(r['case'],r['mode'],r['cache']):r for r in d['rows']};assert len(indexed)==42
        current[side]=indexed;trials[side].append(indexed)
    assert current['baseline'].keys()==current['candidate'].keys()
    for key in current['baseline']:assert current['baseline'][key]['forced_ids']==current['candidate'][key]['forced_ids'],key
    paired.append(current)
rows=[]
for key in sorted(trials['baseline'][0]):
    row=dict(zip(['case','mode','cache'],key));row['forced_id_count']=len(trials['candidate'][0][key]['forced_ids']);row['measurements']={}
    for quantile in ['p50_ns','p95_ns','p99_ns','p999_ns','max_ns']:
        old=statistics.median(p[key]['forcing'][quantile] for p in trials['baseline']);new=statistics.median(p[key]['forcing'][quantile] for p in trials['candidate'])
        row['measurements'][quantile]={'baseline':old,'candidate':new,'ratio':new/max(old,1),'paired_ratios':[p['candidate'][key]['forcing'][quantile]/max(p['baseline'][key]['forcing'][quantile],1) for p in paired]}
    old=statistics.median(p[key]['first_query_ns'] for p in trials['baseline']);new=statistics.median(p[key]['first_query_ns'] for p in trials['candidate']);row['first_query_ns']={'baseline':old,'candidate':new,'ratio':new/max(1,old)};rows.append(row)
summary={'status':'completed','rows':rows,'identical_forced_outputs_all_pairs':True,'timed_queries_per_variant':42*a.pairs*a.repetitions,'notes':['This is forcing-query performance, not normal mask-generation TBM.','cold-state means no source-state full-mask cache; compiled caches may be warm in repeated queries.','Cold first query precedes full-mask reference oracle and is separately reported.','No case is excluded; residual slower cases must be disclosed.','Per-call timer/Python overhead exists in both variants.','Repeated-trial quantiles are not pooled as independent samples.']}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
lines=['# Strict forcing release comparison','',f'{a.pairs} alternating process pairs, {a.repetitions} repeated calls per row. All 42 case/mode/cache outputs match.','','| Case | Mode | Cache | IDs | Baseline p50 us | Candidate p50 us | Ratio |','|---|---|---|---:|---:|---:|---:|']
for r in rows:
    m=r['measurements']['p50_ns'];lines.append(f"| {r['case']} | {r['mode']} | {r['cache']} | {r['forced_id_count']} | {m['baseline']/1000:.3f} | {m['candidate']/1000:.3f} | {m['ratio']:.3f} |")
(a.output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)
