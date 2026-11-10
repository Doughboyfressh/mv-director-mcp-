"""End-to-end run of every tool on synthetic fixtures. Usage: python tests/test_e2e.py /tmp/mvfx"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mv_director import server as S  # noqa: E402

FX = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mvfx")
PROJ = FX / "proj"
OUT = FX / "out"
OUT.mkdir(exist_ok=True)


def show(name, r):
    s = json.dumps(r, indent=1, default=str)
    print(f"\n=== {name} ===\n{s[:1500]}{' ...' if len(s) > 1500 else ''}")


show("check_setup", S.check_setup()["binaries"])
show("inspect", S.inspect_footage([str(FX)])["frame_rates_found"])
show("analyze_song", S.analyze_song(str(FX / "song.wav"), fps=24))
r = S.create_project(str(PROJ), str(FX / "song.wav"), [str(FX)], fps=24, width=640, height=360,
                     title="Test Video", artist="Synth", playback_speeds={"B001": 2.0})
show("create_project problems", r["problems"])
show("takes", r["takes"])
show("conform", S.conform_footage(str(FX / "B001_slowmo_2x.mp4"), str(FX / "B001_slowmo_2x_conformed.mov"),
                                  fps=24, playback_speed=2.0, codec="prores_proxy", project=str(PROJ)))
plan = S.plan_edit(str(PROJ), style="performance", pace=1.0)
show("plan_edit", {k: plan[k] for k in ("total_cuts", "asl_s", "sections", "coverage_gaps", "clip_usage")})
show("cuts", plan["cuts_preview"][:12])
show("export", S.export_to_nle(str(PROJ)))
rc = S.render_rough_cut(str(PROJ), str(OUT / "rough_cut.mp4"), height=360, watermark="FOR REVIEW")
show("rough_cut", rc)
show("storyboard", S.grab_frames("", str(OUT / "storyboard.jpg"), project=str(PROJ), every_cut=True))
show("marry", S.marry_master_audio(str(OUT / "rough_cut.mp4"), str(FX / "song.wav"), str(OUT / "married.mov"),
                                   fade_out=2.0))
show("loudness measure", S.master_loudness(str(OUT / "married.mov")))
show("loudness apply", S.master_loudness(str(OUT / "married.mov"), str(OUT / "married_lufs.mov"), "youtube"))
show("look preview", S.apply_look(str(OUT / "married.mov"), str(OUT / "graded.mov"), preset="teal_orange",
                                  grain=0.3, vignette=0.5, letterbox=2.39, preview_at="40"))
show("look render", S.apply_look(str(OUT / "married.mov"), str(OUT / "graded.mov"), preset="warm_film", letterbox=2.39))
(FX / "lyrics.lrc").write_text("[00:16.50]First verse line here\n[00:20.40]Second line of the verse\n"
                               "[00:32.00]THIS IS THE HOOK\n[00:35.90]sing it back to me\n")
show("lyrics", S.add_lyrics(str(OUT / "graded.mov"), str(FX / "lyrics.lrc"), str(OUT / "lyrics.mov"),
                            uppercase=True, snap_to_beats_project=str(PROJ)))
show("deliverables", S.render_deliverables(str(OUT / "lyrics.mov"), str(OUT / "deliver"),
                                           ["master_prores_hq", "youtube_1080", "vertical_9x16", "client_review"],
                                           title="Test Video", artist="Synth", watermark="CONFIDENTIAL"))
show("cutdowns", S.make_social_cutdowns(str(OUT / "lyrics.mov"), str(OUT / "social"), str(PROJ), [15],
                                        ["vertical_9x16", "instagram_4x5"]))
show("canvas", S.make_spotify_canvas(str(OUT / "lyrics.mov"), str(OUT / "canvas.mp4"), str(PROJ)))
show("qc", S.qc_check(str(OUT / "deliver" / "lyrics_youtube_1080.mp4"), "youtube", str(FX / "song.wav")))
show("thumbs", S.pick_thumbnails(str(OUT / "lyrics.mov"), str(OUT / "thumbs"), 6, str(PROJ)))
show("detect_cuts", {k: v for k, v in S.detect_cuts(str(OUT / "rough_cut.mp4"), project=str(PROJ)).items() if k != "cut_times"})
show("effects plan", S.add_beat_effects(str(OUT / "rough_cut.mp4"), str(OUT / "fx.mp4"), "hype",
                                        [{"type": "nightvision", "where": "section:outro"}], project=str(PROJ),
                                        dry_run=True)["effect_counts"])
show("effects render", S.add_beat_effects(str(OUT / "rough_cut.mp4"), str(OUT / "fx.mp4"), "smooth", project=str(PROJ)))
print("\nALL TOOLS RAN")
