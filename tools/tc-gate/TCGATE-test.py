#!/usr/bin/env python3
"""Usage: python3 TCGATE-test.py. Gate checks for incomplete and altered evidence."""
import importlib.util, pathlib, tempfile, unittest, subprocess, os
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('v',pathlib.Path(__file__).with_name('TCGATE-verdict.py'))
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
def run(epoch='1', ids=None):
    return SimpleNamespace(commit='a'*40, lists={'request':[1,2,3] if ids is None else ids}, starts={'PL':((epoch,'10','abc','def'),1)},problems=[],name='fixture')
class GateTests(unittest.TestCase):
    def test_fresh_equal(self): self.assertEqual(v.part1(run(),run('2'),[]),[])
    def test_same_process(self): self.assertTrue(v.part1(run(),run(),[]))
    def test_changed_token(self): self.assertIn('first difference at token 1',v.part1(run(),run('2',[1,9,3]),[])[0])
    def test_missing_request(self):
        b=run('2');b.lists={};self.assertTrue(v.part1(run(),b,[]))
    def test_missing_start(self):
        b=run('2');b.starts={};self.assertTrue(v.part1(run(),b,[]))
    def test_changed_build(self):
        b=run('2');b.starts['PL']=(('2','10','different','def'),1);self.assertTrue(v.part1(run(),b,[]))
    def test_control_token_difference(self): self.assertTrue(v.part1(run(),run('2',[9,2,3]),[],control=True))
    def test_control_all_changed_still_has_lists(self):
        out=[];bad=v.part1(run(),run('2',[9,8,7]),out,control=True)
        self.assertEqual(len(bad),1)
        self.assertIn('tokens compared 3',out[0])
    def test_control_identical_is_missed(self): self.assertEqual(v.part1(run(),run('2'),[],control=True),[])
    def test_missing_tool_modes(self):
        r=SimpleNamespace(many=[],has_tool=True,problems=[],drive='TOOL prompt-rows exit=0')
        self.assertTrue(v.part2(r,[]))
    def test_failed_tool(self):
        r=SimpleNamespace(many=[],has_tool=True,problems=[],drive='TOOL prompt-rows exit=0\nTOOL nv-tc-groups exit=0\nTOOL dense-tc-rows exit=1')
        self.assertTrue(v.part2(r,[]))
    def test_all_tool_modes(self):
        r=SimpleNamespace(many=[],has_tool=True,problems=[],drive='TOOL prompt-rows exit=0\nTOOL nv-tc-groups exit=0\nTOOL dense-tc-rows exit=0')
        self.assertEqual(v.part2(r,[]),[])
    def test_inventory_missing_read(self):
        r=SimpleNamespace(chat=[],lists={},many=[])
        self.assertTrue(any('8K' in x for x in v.inventory(r)))
    def test_inventory_truncated_cells(self):
        cells=[(s,w,a,0,[1]*255,None,'cell') for s,arms in (('PL',('P',)),('DR',('P','S2','S3')),('DP',('P','S1','S3','S7'))) for w in ('code','prose','structured') for a in arms]
        r=SimpleNamespace(chat=cells,lists={'PL/out-PL-read/receipt.jsonl#g8192d':[1]*160},many=[('group','different',2,[('a',[1]*128),('b',[1]*128)])])
        self.assertEqual(len(v.inventory(r)),24)
    def test_inventory_complete(self):
        cells=[(s,w,a,0,[1]*256,None,'cell') for s,arms in (('PL',('P',)),('DR',('P','S2','S3')),('DP',('P','S1','S3','S7'))) for w in ('code','prose','structured') for a in arms]
        r=SimpleNamespace(chat=cells,lists={'PL/out-PL-read/receipt.jsonl#g8192d':[1]*160},many=[('group','different',2,[('a',[1]*128),('b',[1]*128)])])
        self.assertEqual(v.inventory(r),[])
    def test_runner_without_optional_controls(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d); (root/'kit').mkdir();(root/'runs').mkdir()
            (root/'kit'/'TCGATE-run.sh').write_text('printf "%s\\n" "$*" >> "$VADE_NIGHT_ROOT/calls"\n')
            (root/'kit'/'TCGATE-table.py').write_text('print("stub table")\n')
            env=dict(os.environ,VADE_NIGHT_ROOT=d)
            env.pop('TCGATE_TILE_CONTROL',None);env.pop('TCGATE_SMALL_CONTROL',None)
            check=pathlib.Path(__file__).with_name('TCGATE-check.sh')
            result=subprocess.run(['/bin/bash',str(check),'a'*40,'two','fixture'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len((root/'calls').read_text().splitlines()),7)
            self.assertEqual((root/'runs'/'fixture-table.md').read_text(),'stub table\n')
    def test_empty_run(self):
        with tempfile.TemporaryDirectory() as d: self.assertTrue(v.Run(d).problems)
if __name__=='__main__': unittest.main()
