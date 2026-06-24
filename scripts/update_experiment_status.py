#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MATRIX=ROOT/'experiments/EXPERIMENT_MATRIX.csv'
HISTORY=ROOT/'experiments/status_history.jsonl'
ALLOWED={
 'ready','planned','running','completed','failed','invalid','interrupted',
 'blocked_data','blocked_models','blocked_checkpoint','deferred'
}

def main():
 p=argparse.ArgumentParser(); p.add_argument('experiment_id'); p.add_argument('status',choices=sorted(ALLOWED)); p.add_argument('--reason',required=True); a=p.parse_args()
 with MATRIX.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f)); fields=rows[0].keys()
 matches=[r for r in rows if r['experiment_id']==a.experiment_id]
 if len(matches)!=1: raise SystemExit(f'expected one row for {a.experiment_id}, found {len(matches)}')
 row=matches[0]; old=row['status']; row['status']=a.status
 with MATRIX.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 HISTORY.parent.mkdir(parents=True,exist_ok=True)
 event={'timestamp':datetime.now(timezone.utc).isoformat(),'experiment_id':a.experiment_id,'old_status':old,'new_status':a.status,'reason':a.reason}
 with HISTORY.open('a',encoding='utf-8') as f: f.write(json.dumps(event,ensure_ascii=False)+'\n')
 print(json.dumps(event,indent=2,ensure_ascii=False))
 return 0
if __name__=='__main__': raise SystemExit(main())
