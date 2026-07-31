# Beatless Ambient Generation Recipe

This note records the working baseline for generating slow, atmospheric game ambience with
ACE-Step 1.5. Treat it as an experiment log: preserve successful settings and change one variable
at a time when refining the sound.

## Working Baseline

Use the direct DiT path and leave musical-grid metadata unspecified:

- Set `thinking=false`.
- Leave BPM on Auto / unset.
- Leave key and scale on Auto / unset.
- Leave time signature on Auto / unset.
- Use `lyrics="[Instrumental]"`.
- Use one fixed seed per experiment so prompt changes can be compared.
- Use batch size 1 while testing long pieces.
- Render to FLAC for evaluation and editing.

Write the caption as a positive description of the desired sound. Useful language includes:

- `free-time`, `continuous`, `unmetered`, `sustained`, and `formless`;
- `long held notes`, `overlapping chords`, and `imperceptibly slow harmonic changes`;
- `soft blurred onsets`, `long decays`, `diffuse reverb`, and `non-patterned echoes`;
- `timbral evolution replaces musical events`.

Describe continuous layers instead of conventional song sections. A minimal temporal prompt is:

```text
[Instrumental]

[Continuous free-time drone]

[Sustained pad field changes colour almost imperceptibly]

[Background textures drift through the stereo space]

[Layers thin gradually into a long diffuse decay]
```

## What Did Not Work

The first test enabled the 5 Hz planner and explicitly supplied `50 BPM`, `D minor`, and `4/4`.
It produced percussion and a single repeated beat throughout the piece. A detailed negative prompt
did not prevent that result.

The likely causes were:

1. BPM and meter supplied an explicit rhythmic grid.
2. The 5 Hz planner generated semantic codes with repetitive rhythmic content.
3. XL Turbo does not provide the same classifier-free guidance controls as SFT/base variants.
4. Mentioning unwanted rhythmic elements in the caption may still activate those concepts.

## Successful Test

The successful retry used:

- Model: `acestep-v15-xl-turbo`
- Thinking: disabled
- BPM/key/meter: unspecified
- Duration: 240 seconds
- Inference steps: 8
- Seed: `90317642`
- Caption: positive-only free-time drone and sustained-texture language

The API still reported automatically inferred metadata after generation, but omitting explicit
metadata and planner codes materially improved the audible result.

## Experiment Log

### Environmental pads with rain and insects

- Model: `acestep-v15-xl-turbo`
- Thinking: disabled
- BPM/key/meter: unspecified
- Duration: 300 seconds
- Seed: `46721093`
- Result: More convincingly ambient and beatless, but audibly very dissonant.

Likely prompt causes were overlapping suspended chords, independently moving glass harmonics,
granular haze, and the lack of a stable tonal centre. For the next controlled test, retain the
same seed and generation settings while changing only the harmonic description:

- establish one tonal centre throughout;
- favour perfect fifths, octaves, and a simple major pentatonic palette;
- hold one consonant chord for long spans;
- let existing tones settle before introducing another pitch;
- keep field recordings behind the tonal pad rather than treating them as pitched layers.

## Refinement Method

For each experiment:

1. Start from the working baseline.
2. Keep the seed fixed.
3. Change one prompt concept or parameter.
4. Record whether pulse, percussion, unwanted melody, or abrupt transitions appear.
5. Only adopt a change after comparing the complete render against the baseline.

If XL Turbo repeatedly introduces a pulse, the next model experiment should be XL SFT with its
stronger prompt guidance. Do not combine that model change with a rewritten prompt in the same
test.
