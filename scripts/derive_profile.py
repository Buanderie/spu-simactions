#!/usr/bin/env python3
"""Derive swarmpu-simuav aircraft profiles from PUBLISHED performance, reproducibly.

A profile whose coefficients were typed in by hand is a set of numbers nobody can check, and
for a fleet of real aircraft it is the easiest way to produce a simulation that looks
authoritative and is quietly fictional. So nothing here is a guess: each profile declares
the published figures it is built from, and this script derives the aerodynamic coefficients
that reproduce those figures in simuav's own flight model. Re-run it and you get the same
numbers; change a published figure and the coefficients move with it.

    python3 scripts/derive_profile.py --list
    python3 scripts/derive_profile.py --plane shahed136 --write
    python3 scripts/derive_profile.py --quad mavic3 --write

## The derivation

simuav's `fixedwing.Airframe` is a point-mass-plus-moments model with a parabolic drag
polar, so a published cruise speed and a published rate of climb pin the coefficients:

    1. Stall. At the stall the wing carries the whole weight:
           W = 0.5 rho V_s^2 S CL_max        =>   S = 2W / (rho V_s^2 CL_max)
       with CL_max = CL_0 + CL_alpha * alpha_stall. So a stall speed and a wing area are
       not independent: specifying both over-determines the model, so the area is DERIVED
       from mass, stall speed and CL_alpha, and the span is published.

    2. Cruise drag. Level flight at V_c needs thrust equal to drag, and drag is
           D = 0.5 rho V_c^2 S (CD_0 + CL_c^2 / (pi AR e))
       with CL_c = 2W / (rho V_c^2 S). The weight is therefore the level-flight thrust,
       and CL_c is an OUTPUT of (1), not an input.

    3. CD_0. Not derivable from cruise alone -- any CD_0 with the matching induced term
       flies the same cruise. It is set from the published glide ratio / L/D, which is
       the number that actually distinguishes one airframe from another, and the induced
       term is what makes the polar consistent with it.

4. Thrust available. Excess power at the published climb rate fixes it. In a steady
    climb at speed V_c with rate V_z the thrust must both beat drag and lift the weight:
           T = D + W sin(gamma),   sin(gamma) = V_z / sqrt(V_c^2 + V_z^2)
       then THRUST_MAX is rounded up with margin so the model's *maximum* exceeds the
       published climb rather than exactly matching it, because a model that can only just
       reach its quoted performance is useless for testing.

    5. Published engine rating, when there is one. Small turbojets are the one powerplant
       whose thrust is published as a plain number -- Telefly rates the JT80 at 80 kgf and
       the TF-TJ2000A at 200 kgf, and HUR publishes both figures with the airframe. That
       is a stronger input than a climb rate nobody has measured, so a spec may give
       `thrust_n` instead of `climb_ms`: THRUST_MAX becomes the published rating and the
       climb rate becomes the OUTPUT, obtained by inverting step 4. It also produces a
       real consistency check -- if the published thrust cannot even beat cruise drag at
       the published cruise speed, the inputs contradict each other and the profile is
       refused rather than flown.

Two further inputs are estimates and are labelled as such in every profile that uses them,
because nothing publishes them: the stall speed (nobody publishes a stall speed for a
loitering munition) and, for the jet types, the lift curve. Deltas and cruise-missile
layouts do not carry CL_max 1.68 at 16 deg; a spec may therefore override the lift curve
through its `aero` block -- `cl_alpha`, `cl0`, `alpha_stall_deg`, `e_oswald` -- and the
wing area that falls out of the derivation changes with it.

Everything a profile asserts is echoed into the JSON: the published inputs, the derived
coefficients, and a `confidence` that says which are which. A profile fitted to published
numbers is `published`. It is NOT `measured`, and the difference matters: nobody has flown
this rig and watched a Shahed climb at 4.6 m/s.
"""
import argparse
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RHO = 1.225          # kg/m^3, sea level; simuav's own level
G = 9.80665


# ------------------------------------------------------------------ published specs
#
# Aircraft actually fielded in the Ukraine conflict and other recent deployments, with the
# figures below taken from manufacturer datasheets and open-source defence analysis. Every
# entry records where each number came from in `sources`. They are PUBLISHED performance,
# not measurements taken here, and profiles built from them are labelled `published`:
# good enough to fly a guidance law against, not good enough to certify an airframe.

PLANES = {
    # name: (title, mass_kg, wing_span_m, stall_ms, cruise_ms, climb_ms, sources)
    "shahed136": dict(
        title="Shahed-136 one-way attack UAV",
        mass_kg=200.0, wing_span_m=2.5, stall_ms=28.0, cruise_ms=51.4, climb_ms=4.6,
        max_speed_ms=51.4,
        length_m=3.5, ld_claimed=6.0,
        sources=["HESA / Wikipedia / CSIS Missile Threat: 200 kg MTOW, 3.5 m long, 2.5 m "
                 "wingspan, ~185 km/h, 2,000-2,500 km range, Mado MD-550 piston engine "
                 "(37 kW), 30-50 kg warhead",
                 "The 135 kg figure that circulates for the Shahed-136 is the SHAHED-131's "
                 "weight; it is not used here",
                 "ESTIMATED, not published: stall 28 m/s, which is what an ~82 kg/m^2 wing "
                 "loading on a 2.5 m span delta gives (2.4 m^2 of wing). It also puts the "
                 "cruise at 0.50 CL, about 30% of CL_max, which is where a real delta OWA "
                 "cruises. A stall speed near 17 m/s -- which is what an earlier version of "
                 "this profile used -- implies 4.5 m^2 of wing on a 2.5 m span, i.e. an "
                 "aspect ratio of 1.4, and a cruise CL of 0.18: a sailplane that happens to "
                 "be a bomb",
                 "Sanity check on the thrust: MD-550 power is published at 37 kW, which is "
                 "~720 N of thrust at 51.4 m/s; the derived 580 N is 20% under it, so the "
                 "profile is conservative rather than optimistic"],
        notes="The most widely used one-way attack UAV of the conflict. Public performance "
              "figures are broad and partly contested; treat this as the right order of "
              "magnitude, not a type certificate. Derivatives (Geran-2, and the "
              "Shahed-136JR recon variant) differ mainly in payload and warhead."),
    "geran2": dict(
        title="Geran-2 (Shahed-136 in Russian service)",
        mass_kg=220.0, wing_span_m=2.5, stall_ms=29.5, cruise_ms=51.4, climb_ms=4.6,
        max_speed_ms=51.4,
        length_m=3.5, ld_claimed=6.0,
        sources=["CSIS Missile Threat names the Geran-2 as the Shahed-136 and the Geran-1 "
                 "as the Shahed-131; open-source defence analysis puts Russian production at "
                 "'200 kg +' with a 50 kg HE or 90 kg thermobaric/fragmentation warhead",
                 "220 kg here rather than 200 kg: the E-series Geran-2s that carry the 90 kg "
                 "warhead are the heavier build, and the wing area and stall speed follow "
                 "from that mass",
                 "drone-warfare.com quotes ~350 kg for the Geran-2; that is NOT used. It "
                 "contradicts the Geran-2 = Shahed-136 identity that CSIS and OSMP both "
                 "state, and the Shahed-136's own 200 kg figure is the better sourced one"],
        notes="Same airframe and powerplant as the Shahed-136 at higher gross mass: the "
              "derived wing area and stall speed follow from the mass, which is exactly the "
              "kind of coupling that a hand-typed coefficient gets wrong."),
    "geran3": dict(
        title="Geran-3 (jet-powered Shahed-238 derivative)",
        mass_kg=380.0, wing_span_m=3.0, stall_ms=35.0, cruise_ms=83.0, thrust_n=785.0,
        max_speed_ms=91.7,
        length_m=3.5, ld_claimed=6.0, confidence="estimated",
        aero=dict(cl_alpha=4.6, alpha_stall_deg=13.0, e_oswald=0.78),
        sources=["Ukrainian Defence Intelligence (HUR), 'U' series analysis via Militarnyi: "
                 "Chinese Telefly JT80 turbojet, flight speed 300-370 km/h, range up to "
                 "1,000 km, top speed reached in air-defence zones and the terminal dive",
                 "UNITED24 Media / Militarnyi: ~3.5 m long, ~3 m wingspan, 380 kg takeoff "
                 "weight",
                 "Telefly JT80: ~80 kgf (~785 N) per the manufacturer (OSMP analyst note, "
                 "SecurityLab review). That rating is used as THRUST_MAX and the climb rate "
                 "is derived from it: 3.8 m/s at 83 m/s is all 785 N buys over a 619 N cruise "
                 "drag. An earlier version of this profile guessed the climb instead and "
                 "then had to be checked against the engine",
                 "Russian claims of 500-600 km/h (and a Czech PBS TJ150 engine) are "
                 "unverified and NOT used here",
                 "ESTIMATED, not published: stall 35 m/s (~127 kg/m^2 on the 3.0 m span); "
                 "cruise 83 m/s = 300 km/h, the low end of HUR's band; L/D 6 (Shahed-136 "
                 "family). The lift curve is a delta's, not a wing's: CL_max 1.16 at 13 deg "
                 "and Oswald 0.78"],
        notes="Size, mass and speed have a public source and so does the engine rating; the "
              "stall speed and the polar are estimates from the Shahed-136 family and the "
              "change of powerplant. Good enough to ask how the law copes with a ~90 m/s "
              "delta-wing munition; not a description of the real aircraft."),
    "geran4": dict(
        title="Geran-4 (jet-powered, reinforced delta OWA UAV)",
        mass_kg=450.0, wing_span_m=3.0, stall_ms=38.0, cruise_ms=88.9, thrust_n=1600.0,
        max_speed_ms=138.9,
        length_m=3.5, ld_claimed=6.0, confidence="estimated",
        aero=dict(cl_alpha=4.6, alpha_stall_deg=13.0, e_oswald=0.78),
        sources=["HUR via Militarnyi / defence-industry.eu (25 May 2026): new reinforced "
                 "airframe with the wings permanently merged into the centre section, 3.5 m "
                 "long, 3.0 m wingspan, 450 kg MTOW, 50 kg OFZBCH-50/TBBCH-50M warhead or a "
                 "90 kg TBBCH-90, 500 km/h max, 5,000 m ceiling, 450 km range",
                 "HUR: 350 km/h max, 320 km/h cruise, and active manoeuvre at 300-400 km/h",
                 "HUR identified two Telefly engines: LX-WP-160 at 160 kgf (1,600 N) and the "
                 "TF-TJ2000A at 200 kgf (1,960 N) seen on the Geran-5. The weaker 1,600 N "
                 "engine is used here so the profile is not the best-case airframe",
                 "ESTIMATED, not published: stall 38 m/s (~151 kg/m^2 on the 3.0 m span); "
                 "L/D 6; the delta lift curve, as for the Geran-3",
                 "Cross-check on the geometry: 1,600 N of thrust balances the derived drag at "
                 "~139 m/s, which is HUR's published 500 km/h maximum. A wing area ~50% "
                 "larger would have put the top speed at 700 km/h, which this engine does "
                 "not have"],
        notes="The middle member of the jet Gerans and the one Ukraine's interceptor drones "
              "shoot down most often: it is slow enough to chase (320 km/h cruise against "
              "the STING's 343 km/h) and fast enough that a piston Geran-2 is out of the "
              "question."),
    "geran5": dict(
        title="Geran-5 (jet-powered, cruise-missile-like OWA UAV)",
        mass_kg=850.0, wing_span_m=3.2, stall_ms=53.0, cruise_ms=125.0, thrust_n=1960.0,
        max_speed_ms=166.7,
        length_m=6.5, ld_claimed=8.0, confidence="estimated",
        sources=["HUR specification card on the War & Sanctions portal (19 Jan 2026), built "
                 "from an intact wreck: 6.5 m long, 3.2 m wingspan, 850 kg MTOW, 90 kg BCh-90 "
                 "warhead, 450-600 km/h cruise, 950 km estimated range, 2 h endurance, 6,000 m "
                 "ceiling, combat use recorded at 200 m - 3 km",
                 "HUR's earlier announcement (11 Jan 2026) gave ~6 m long and 'up to 5.5 m' "
                 "wingspan. THE CARD'S 3.2 m IS USED HERE: it comes from the wreckage, and "
                 "5.5 m on 850 kg means 6.6 m^2 of wing, an aspect ratio of 4.6 and a CD_0 of "
                 "0.015 -- cleaner than a clean airliner. 3.2 m gives ~2.9 m^2, 280 kg/m^2 "
                 "and a CD_0 of 0.027, which is what a jet cruise missile with a dorsal "
                 "intake and a tail actually is",
                 "Ukrainian Defence Intelligence (HUR) via Kyiv Independent / RFE/RL / Kyiv "
                 "Post: Chinese Telefly TF-TJ2000A turbojet, 450-600 km/h, range ~950 km, "
                 "ceiling 6 km, 90 kg warhead; can be air-launched from a Su-25",
                 "Euromaidan Press (2026-09-29): intercepted over Chernihiv at 425 km/h, "
                 "above 3 km; first Ukrainian interceptor kills 19-22 Sept 2026",
                 "TELEFLY TF-TJ2000A: 200 kgf (1,960 N), 29.24 kg, 1.30 kg/kgf/h, Mach 0.95 "
                 "max, from the manufacturer's own archived Made-in-China listing. That rating "
                 "is used as THRUST_MAX; the 13.8 m/s climb is what 1,960 N buys over a "
                 "1,043 N cruise drag at 125 m/s",
                 "ESTIMATED, not published: stall 53 m/s (~290 kg/m^2 on the 3.2 m span); "
                 "L/D 8 (drone-warfare.com's own 6-10 assumption, used as a mid value); "
                 "cruise 125 m/s = 450 km/h, the low end of HUR's band and consistent with "
                 "the 425 km/h intercept"],
        notes="A conventional-layout jet, closer to a small cruise missile than to the "
              "Shahed family. Two published span figures are in circulation and they are "
              "not compatible with each other; this profile takes the wreckage one and the "
              "polar that comes with it. At 125 m/s and 35 deg of bank its turn radius is "
              "~2.3 km, so the heading it starts with matters far more than for anything "
              "else in the fleet. Size, mass, speed and thrust are reported; the wing area "
              "and the polar are estimates."),
    "karrar": dict(
        title="HESA Karrar (Iranian jet-powered strike/recon UAV)",
        mass_kg=700.0, wing_span_m=3.0, stall_ms=49.0, cruise_ms=222.0, climb_ms=20.0,
        max_speed_ms=250.0,
        length_m=5.5, ld_claimed=9.0, confidence="estimated",
        sources=["Iranian announcement (2010) via defense-update.com and army-technology: "
                 "1,000 km range, 900 km/h speed, 500 kg payload class; GlobalSecurity puts "
                 "the Karrar at ~0.7 t",
                 "Design lineage: the American Beechcraft BQM-126 target drone, 5.51 m long, "
                 "3 m span, ~0.6 t loaded, ~950 km/h, >2 h endurance; HUR's wreckage "
                 "examination found the Geran-5 'resembles the Iranian Karrar' and "
                 "drone-warfare.com traces it back to the same airframe",
                 "Powerplant: Toloue, an Iranian derivative of the French Microturbo TRI 60. "
                 "Static thrust is NOT published for the Karrar's engine (the 4.4 kN "
                 "Toloue-5 figure belongs to Iran's anti-ship missile programme), so thrust "
                 "is derived from the climb rate as usual",
                 "ESTIMATED, not published: stall 49 m/s (~250 kg/m^2), cruise 222 m/s = "
                 "800 km/h (below the 900-950 km/h top speed, which is what a cruise figure "
                 "should be), climb 20 m/s, L/D 9"],
        notes="The Iranian design the Geran-5 is a copy of, and a useful counterfactual for "
              "it: same layout class, 1,400 N of thrust rather than 1,960 N, and it is "
              "landing-capable rather than rail-launched. Unused by Russia as far as is "
              "public, but it is the airframe the Geran-5 was derived from."),
    "lancet3": dict(
        title="ZALA Lancet-3 (Item 51) loitering munition",
        mass_kg=12.0, wing_span_m=2.4, stall_ms=13.0, cruise_ms=30.6, climb_ms=3.0,
        max_speed_ms=83.3,
        length_m=1.7, ld_claimed=7.0, confidence="estimated",
        sources=["ZALA Aero / Wikipedia: 12 kg MTOW, 3 kg warhead, 40 km range, 40 min "
                 "endurance, 80-110 km/h cruise, 300 km/h terminal dive, electric motor, "
                 "catapult-launched with a parachute recovery. Heavily used in Ukraine "
                 "against artillery and air defences",
                 "Open-source dimensions: 1.7 m long, 2.4 m span (globalmilitary.net); "
                 "GlobalSecurity gives ~1.65 m and a ~1.0 m span, which is the older "
                 "Lancet",
                 "ESTIMATED, not published: stall 13 m/s (~17 kg/m^2 on the 2.4 m span, i.e. "
                 "~0.70 m^2 of wing), climb 3 m/s, L/D 7. The derived 33 N of cruise thrust "
                 "on a 12 kg airframe is about the 2 kW a Lancet-class electric motor has "
                 "to give at 30 m/s"],
        notes="The smallest aircraft in the fleet and the one that has destroyed the most "
              "hardware per unit in this war: 12 kg, launched from a truck, 40 km of range. "
              "It is also a speed case the fleet otherwise lacks -- it cruises at a third "
              "of a Shahed-136's speed and dives at three times it."),
    "kemankes1": dict(
        title="Baykar Kemankes 1 (jet-powered mini cruise missile)",
        mass_kg=45.0, wing_span_m=1.14, stall_ms=41.0, cruise_ms=72.0, climb_ms=15.0,
        max_speed_ms=91.7,
        length_m=1.73, ld_claimed=8.0, confidence="estimated",
        sources=["Wikipedia / Baykar: 45 kg MTOW, 1.73 m long, 1.14 m span, 10 kg payload, "
                 "100 km operational range, 20 min boost, 180 kn (330 km/h) maximum, jet "
                 "engine, launched from TB2/TB3/Akinci. First launched from a TB2 in "
                 "March 2025",
                 "ESTIMATED, not published: stall 41 m/s (~173 kg/m^2 on the 1.14 m span, "
                 "~0.26 m^2 of wing), cruise 72 m/s = 260 km/h (the published 330 km/h is "
                 "a maximum, not a cruise), climb 15 m/s, L/D 8"],
        notes="A released weapon rather than a launched one: it is a 45 kg cruise missile "
              "that starts life under the wing of an MALE, which is the one way an aircraft "
              "of this size reaches a target with no launch infrastructure at all. The "
              "derived 135 N is a low thrust-to-weight jet, so this profile is generous "
              "about how slowly it would accelerate."),
    "pd2": dict(
        title="UkrSpecSystems PD-2 tactical UAS",
        mass_kg=55.0, wing_span_m=5.0, stall_ms=18.06, cruise_ms=27.78, climb_ms=3.0,
        max_speed_ms=38.9,
        length_m=2.85, ld_claimed=14.0,
        sources=["UkrSpecSystems' own PD-2 datasheet: 55 kg MTOW, 5 m wingspan, 2.85 m long, "
                 "1.02 m high, 11 kg payload, 100 km/h cruise, 140 km/h max, 65 km/h STALL "
                 "SPEED, 3 m/s max climb, 3,800 m ceiling, 8+ h endurance, catapult or VTOL "
                 "take-off, parachute or runway recovery",
                 "Glide ratio 14-19 published by UkrSpecsystems; 14 is used, the low end",
                 "This is the best-documented airframe in the fleet: the stall speed is "
                 "PUBLISHED, so nothing here is estimated except the polar's L/D"],
        notes="One of the most-produced Ukrainian reconnaissance UAVs and the only profile "
              "here whose stall speed comes from a datasheet rather than from a wing-loading "
              "argument. The derivation puts 1.6 m^2 on a 5 m span (aspect ratio 15) at 34 "
              "kg/m^2, which is what the published 65 km/h stall implies, and the resulting "
              "138 N of thrust is what a 100 cc engine gives at 28 m/s."),
    "shark": dict(
        title="UkrSpecSystems Shark UAS",
        mass_kg=13.5, wing_span_m=3.4, stall_ms=16.67, cruise_ms=20.83, climb_ms=2.5,
        max_speed_ms=36.1,
        length_m=1.65, ld_claimed=12.0,
        sources=["UkrSpecsystems' own Shark datasheet: up to 13.5 kg MTOW, 3.4 m wingspan, "
                 "1.65 m long, 0.3 m high, 75 km/h cruise, 130 km/h max, 60 km/h STALL "
                 "SPEED, 4 h flight time, 3,000 m ceiling, electric, catapult and parachute, "
                 "80 km encrypted datalink",
                 "L/D 12 for an electric high-aspect-ratio airframe of this class",
                 "Like the PD-2 this is a datasheet profile, stall speed included"],
        notes="A small electric reconnaissance UAV with an unusually high aspect ratio for "
              "its mass: the published 60 km/h stall on a 3.4 m span puts 0.46 m^2 of wing "
              "on it, i.e. aspect ratio 25, and a cruise CL of 1.07. That is what an "
              "efficient electric wing looks like, and it is why the derived CD_0 is high "
              "(0.09) even though the induced term is almost all of the glide ratio."),
    "bayraktar_mini": dict(
        title="Bayraktar Mini B hand-launched mini UAV",
        mass_kg=5.8, wing_span_m=2.45, stall_ms=11.0, cruise_ms=17.0, climb_ms=2.0,
        max_speed_ms=18.1,
        length_m=1.25, ld_claimed=12.0, confidence="estimated",
        sources=["Baykar's own product page: MTOW 5.8 kg, 580 g payload, 2.45 m wingspan, "
                 "1.25 m long, 30-35 KTAS maximum speed, 100 min endurance, 8,000 ft "
                 "service ceiling, 3,000 ft operational altitude, hand launch, parachute or "
                 "belly landing. Used by Ukraine since 2022",
                 "CONFLICTING FIGURES, and the newer manufacturer page is used here: older "
                 "sources (Wikipedia, Army Technology) give a Mini B of 4.5 kg, 1.9 m span "
                 "and 110 km/h cruise, and an Army Technology figure of 60 km/h cruise "
                 "agrees with the 30-35 kt maximum better than the 110 km/h does",
                 "ESTIMATED, not published: stall 11 m/s (~13 kg/m^2 on the 2.45 m span), "
                 "climb 2 m/s, L/D 12, cruise 17 m/s = 61 km/h"],
        notes="The smallest fixed-wing aircraft here and the reason a 5 kg UAV needs its own "
              "profile: the wing area the derivation produces (0.46 m^2, aspect ratio 13) is "
              "sailplane territory, and a guidance law tuned on a Shahed will stall out of "
              "the envelope here first."),
    "switchblade300": dict(
        title="Switchblade 300 loitering munition",
        mass_kg=13.6, wing_span_m=0.69, stall_ms=26.0, cruise_ms=48.6, climb_ms=4.0,
        max_speed_ms=48.6,
        length_m=1.3, ld_claimed=6.0,
        sources=["Switchblade 300 Block 20 datasheet: 13.6 kg, 0.69 m wingspan, 175 km/h "
                 "cruise, 60+ km range, 15 min loiter, 30 min flight",
                 "ESTIMATED, not published: stall 26 m/s, i.e. ~0.20 m^2 of wing at ~68 "
                 "kg/m^2, which is the wing loading a 0.69 m span tube-launched munition "
                 "has. The 15 m/s an earlier version of this profile used implied 0.58 m^2 "
                 "on a 0.69 m span -- an aspect ratio of 0.8, i.e. a disc"],
        notes="Originally a hand-launched reconnaissance UAS, now fielded as a one-way "
              "attack munition. Small, fast and low: the 48.6 m/s cruise against a 13.6 kg "
              "mass is the demanding part, and it is what makes the guidance's speed "
              "schedule matter more here than on a heavy aircraft."),
    "switchblade600": dict(
        title="Switchblade 600 Block 20 loitering munition",
        mass_kg=22.7, wing_span_m=1.30, stall_ms=22.5, cruise_ms=51.1, climb_ms=4.0,
        max_speed_ms=51.1,
        length_m=2.0, ld_claimed=6.0,
        sources=["Switchblade 600 datasheet: 22.7 kg, 184 km/h max, 40+ km range, "
                 "20+ min endurance, 130 m altitude capability",
                 "ESTIMATED, not published: stall 22.5 m/s (~54 kg/m^2, ~0.42 m^2 of wing)"],
        notes="Armed and deployed by ramp, so it arrives with no runway and no catapult. "
              "The rig's catapult launch is a fair stand-in for a ramp drop."),
    "bayraktar_tb2": dict(
        title="Bayraktar TB2 MALE UCAV",
        mass_kg=700.0, wing_span_m=12.2, stall_ms=25.0, cruise_ms=67.1, climb_ms=4.0,
        max_speed_ms=61.7,
        length_m=6.7, cd0=0.028, ld_claimed=12.0,
        sources=["Baykar TB2 published figures: 700 kg MTOW, 12.2 m wingspan, 220 km/h max "
                 "speed, 27 h endurance, 150 kg payload, 15,000 ft service ceiling",
                 "Aerospace Middle East / Janes analyses of TB2 cruise performance"],
        notes="A large MALE aircraft, not a loitering munition: the useful thing about it "
              "here is that it is a genuinely different regime -- heavy, slow-turning, "
              "long-range -- which is where a speed-schedule-based law is most likely to "
              "be wrong."),
    "tb3": dict(
        title="Bayraktar TB3 heavy MALE UCAV",
        mass_kg=1300.0, wing_span_m=12.3, stall_ms=26.0, cruise_ms=70.0, climb_ms=3.5,
        max_speed_ms=69.4,
        length_m=8.0, cd0=0.028, ld_claimed=12.0,
        sources=["Baykar TB3 published figures: ~1300 kg MTOW, 12.3 m wingspan, up to 19 h "
                 "endurance with wingtip camera pods"],
        notes="TB3 is the same airframe enlarged for a sensor payload; the wingtip pods "
              "add drag that this profile does not model, so cruise is optimistic."),
    "akinci": dict(
        title="Bayraktar Akinci (UAV-3) MALE UCAV",
        mass_kg=1350.0, wing_span_m=12.3, stall_ms=27.0, cruise_ms=80.0, climb_ms=5.0,
        max_speed_ms=90.0,
        length_m=10.8, cd0=0.027, ld_claimed=13.0,
        sources=["Baykar UAV-3 published figures: 1350 kg MTOW, 12.3 m wingspan, 324 km/h "
                 "max speed, 20 h+ endurance, 300 kg payload, armed with MAM-L"],
        notes="The fastest aircraft in this fleet. At 80 m/s cruise its roll rate and turn "
              "radius, not its speed, are what a flyby law has to respect, and the "
              "reference plant's 1.2 s horizontal time constant is almost certainly too "
              "slow for it."),
    "orlan10": dict(
        title="STC Orlan-10 tactical reconnaissance UAS",
        mass_kg=16.0, wing_span_m=3.1, stall_ms=17.0, cruise_ms=30.0, climb_ms=2.0,
        max_speed_ms=41.7,
        length_m=2.0, ld_claimed=12.0,
        sources=["HUR War & Sanctions card and GlobalSecurity: 16-18 kg MTOW, 3.1 m "
                 "wingspan, 1.8-2.0 m long, 90-150 km/h, 5.5-6 kg payload, up to 16 h "
                 "endurance, 5,000 m ceiling, Saito FA-62B four-stroke piston, catapult and "
                 "parachute",
                 "The 14 m wingspan that circulates for the Orlan-10 belongs to the "
                 "ORLAN-30 (STC's sailplane, a different and much larger aircraft). An "
                 "earlier version of this profile carried it, which put the aspect ratio at "
                 "292 and the derived wing area at 0.67 m^2 -- the kind of number that looks "
                 "plausible in a table and describes nothing",
                 "ESTIMATED, not published: stall 17 m/s (~30 kg/m^2 on the 3.1 m span, "
                 "aspect ratio 18), climb 2 m/s, L/D 12"],
        notes="A 16 h endurance hand-launched reconnaissance aircraft, and the cheapest "
              "thing in the fleet per hour of coverage. The high aspect ratio for its mass "
              "means the induced-drag term dominates its polar, which is the point: it is "
              "the aircraft that has to keep flying after everyone else has run out of "
              "battery."),
    "mq1_predator": dict(
        title="General Atomics MQ-1 Predator",
        mass_kg=1020.0, wing_span_m=14.0, stall_ms=22.0, cruise_ms=81.0, climb_ms=4.0,
        max_speed_ms=60.3,
        length_m=6.9, cd0=0.028, ld_claimed=10.0,
        sources=["MQ-1 Predator flight manual and published performance: 1020 kg MTOW, "
                 "14 m wingspan, 217 km/h max speed, 15+ h endurance, pusher prop"],
        notes="Older but still the reference MALE shape, and one where the published "
              "numbers are good enough to check the derivation against: pusher propeller, "
              "so the control response in simuav's model is at least the right sign."),
}

QUADS = {
    # name: (title, mass_kg, wheelbase_diagonal_m, hover_thrust_per_rotor_n, sources)
    "mavic3": dict(
        title="DJI Mavic 3 Enterprise",
        mass_kg=0.915, wheelbase_m=0.795, n_rotors=4, thrust_per_rotor_n=3.6,
        max_speed_mps=15.83,
        sources=["DJI Mavic 3 Enterprise published specifications: 915 g, 795 mm "
                 "diagonal wheelbase, 57 km/h max, 43 min flight, 21 min at 5000 m"],
        notes="Used on both sides of the conflict in quantity. The published 57 km/h "
              "(15.8 m/s) matches the iris-class figure this repo already had, which is "
              "why the quad guidance numbers transfer without retuning."),
    "mavic2ent": dict(
        title="DJI Mavic 2 Enterprise Advanced",
        mass_kg=2.100, wheelbase_m=0.610, n_rotors=4, thrust_per_rotor_n=8.0,
        max_speed_mps=15.83,
        sources=["DJI Mavic 2 Enterprise Advanced published specifications: 2100 g, 610 mm "
                 "diagonal wheelbase, 57 km/h max, 31 min flight"],
        notes="Heavier and larger-rotored than the Mavic 3 for a similar top speed, so it "
              "climbs and accelerates better; the mass and wheelbase are what make the "
              "difference and both are published."),
    "mini4pro": dict(
        title="DJI Mini 4 Pro",
        mass_kg=0.249, wheelbase_m=0.400, n_rotors=4, thrust_per_rotor_n=1.1,
        max_speed_mps=5.0,
        sources=["DJI Mini 4 Pro published specifications: 249 g, 400 mm diagonal "
                 "wheelbase, 18 km/h in S mode, 34 min flight"],
        notes="Sub-250 g and deliberately slow. Its minimum flyable speed is the "
              "interesting number: this is an airframe where 'arrive as slowly as possible' "
              "is genuinely limited, and it is the one the intercept floor competes with."),
    "matrix350rtk": dict(
        title="DJI Matrice 350 RTK heavy-lift quad",
        mass_kg=6.470, wheelbase_m=1.000, n_rotors=4, thrust_per_rotor_n=26.0,
        max_speed_mps=23.0,
        sources=["DJI Matrice 350 RTK published specifications: 6470 g with payload, "
                 "1.0 m diagonal wheelbase, 23 m/s max, 56 min flight, 6 kg payload"],
        notes="A lifting airframe rather than a camera one: 6.5 kg and 26 N per rotor. It "
              "is here because it changes the VERTICAL regime by a factor of five over the "
              "iris, which is the parameter that closed this repo's live blocker."),
    "rotem_l": dict(
        title="IAI Rotem L quadrotor loitering munition",
        mass_kg=6.5, wheelbase_m=0.63, n_rotors=4, thrust_per_rotor_n=37.5,
        max_speed_mps=25.0,
        confidence="estimated",
        sources=["IAI / Wikipedia: 6.5 kg, 1.2 kg grenade warhead, 36-43 km/h (10-12 m/s) "
                 "cruise, 90 km/h terminal dive, 10 km range, 30 min endurance with the "
                 "lethal payload, <1 m CEP, 91-305 m operating altitude. In IDF service "
                 "since 2016, and an export option for counter-UAS packages"],
        notes="The first rotary-wing loitering munition and still the reference one. The "
              "mass and the speeds are the manufacturer's; the wheelbase (0.63 m) and the "
              "rotor thrusts (37.5 N, a 2.35:1 thrust-to-weight) are read off photographs, "
              "which is why this one is "
              "`estimated` rather than `published`. Its value here is the terminal regime: "
              "it arrives at 25 m/s, which is faster than any multirotor in the fleet "
              "cancels at."),
    "rotem_alpha": dict(
        title="IAI Rotem Alpha quadrotor loitering munition",
        mass_kg=25.0, wheelbase_m=1.00, n_rotors=4, thrust_per_rotor_n=135.0,
        max_speed_mps=18.0,
        confidence="estimated",
        sources=["IAI / Wikipedia: 25 kg, 60 min airborne endurance or 24 h in a perch and "
                 "stare mode, 40 km range, man-portable, unveiled 2023",
                 "ESTIMATED, not published: wheelbase 1.0 m and 135 N per rotor (a 2.2:1 "
                 "thrust-to-weight), read off "
                 "photographs; the 25 kg and the endurance are the manufacturer's"],
        notes="The same airframe idea as the Rotem L at four times the mass. It is the "
              "useful comparison for this repo: same guidance, same 2.2:1 thrust-to-weight, "
              "and a vertical authority four times larger, which is the parameter that "
              "decides whether a loitering munition can be stopped by a fast mover."),
    "fpv8in": dict(
        title="8-inch tactical FPV quadcopter (class profile)",
        mass_kg=2.6, wheelbase_m=0.62, n_rotors=4, thrust_per_rotor_n=15.0,
        max_speed_mps=47.2,
        confidence="estimated",
        sources=["Class profile, not one airframe. Figures are those published for a "
                 "current 8-inch tactical FPV strike drone (Woot-Tech Dark Phoenix class: "
                 "1.0-2.0 kg payload, 120 km/h cruise, 170 km/h max, 25 km range with "
                 "1.0 kg of it, 3,000 m ceiling)",
                 "ESTIMATED, not published: 2.6 kg all-up, 0.62 m wheelbase and 15 N per "
                 "rotor, which is a 2.35:1 thrust-to-weight on 8-inch rotors -- what an "
                 "8S/6S tactical build actually flies at"],
        notes="The most numerous aircraft in this war and, deliberately, the vaguest profile "
              "in the fleet: an FPV strike drone is a class of parts bin, not a type. What "
              "matters here is that it is CHEAP and it is FAST (33 m/s cruise, 47 m/s "
              "terminal) at a range of a few km, which is a different intercept problem from "
              "a Shahed at 51 m/s from 2,000 km away."),
}


# ------------------------------------------------------------------------ derivation

# Aspect ratios a real winged aircraft occupies. A value outside this band is not an unusual
# aeroplane, it is a contradiction between the published inputs -- and the derivation is
# built so that contradictions show up HERE rather than as a simulation that flies like
# something from another planet.
AR_BAND = (0.8, 40.0)


def consistency(spec, d, derived):
    """Checks on the derived coefficients, with the published inputs that produced them."""
    problems = []
    ar = derived["aspect_ratio"]
    if not (AR_BAND[0] <= ar <= AR_BAND[1]):
        problems.append(
            f"aspect ratio {ar:.0f} is outside the {AR_BAND[0]}-{AR_BAND[1]} band any real "
            f"winged aircraft occupies: mass {spec['mass_kg']} kg, stall "
            f"{spec['stall_ms']} m/s and span {spec['wing_span_m']} m cannot all be true "
            f"at once (they imply {d['wing_area_m2']:.2f} m^2 of wing)")
    if d["cd0"] >= d["cl0"] * 0.6:
        problems.append(f"CD_0 {d['cd0']:.4f} is implausibly large against CL_0 "
                        f"{d['cl0']:.2f}")
    if derived["cl_cruise"] > derived["cl_max"] * 1.02:
        problems.append(f"cruise CL {derived['cl_cruise']:.2f} exceeds CL_max "
                        f"{derived['cl_max']:.2f}: the aircraft would be stalled in cruise")
    # A published engine rating is checkable against the published cruise speed, and when the
    # two disagree it is always the ENGINE that is wrong, not the drag polar. This is the
    # check that catches an over-generous wing: halve the area and the drag halves, and a
    # 785 N Telefly suddenly "supports" a cruise it could not.
    if spec.get("thrust_n") and derived["thrust_margin_over_cruise_drag"] < 1.0:
        problems.append(
            f"the published {spec['thrust_n']:.0f} N of thrust cannot sustain level flight "
            f"at {spec['cruise_ms']:.0f} m/s: the derived cruise drag is "
            f"{derived['cruise_drag_n']:.0f} N. The wing area ({d['wing_area_m2']:.2f} m^2) "
            f"or the stall speed ({spec['stall_ms']} m/s) is wrong")
    # No "thrust exceeds weight" check: it is wrong for anything fast. A jet-powered MALE
    # cruising at 80 m/s carries a thrust-to-weight of about 0.16 and that is correct --
    # drag, not weight, is what it is fighting. The weight is only the dominant term on a
    # slow aircraft.
    return problems


def derive_plane(spec, cl0=0.28, cl_alpha=5.0, e_oswald=0.80, cl_elevator=0.08,
                 alpha_stall_deg=16.0, margin=1.15):
    """simuav `Airframe` coefficients from the published performance."""
    W = spec["mass_kg"] * G
    alpha_stall = math.radians(alpha_stall_deg)
    cl_max = cl0 + cl_alpha * alpha_stall
    # (1) wing area from mass, stall speed and CL_max.
    S = 2.0 * W / (RHO * spec["stall_ms"] ** 2 * cl_max)
    b = spec["wing_span_m"]
    AR = b * b / S
    # (2) cruise: weight is the level-flight thrust, CL_cruise falls out of (1).
    Vc = spec["cruise_ms"]
    cl_c = 2.0 * W / (RHO * Vc * Vc * S)
    cd_i = cl_i_cd(cl_c, AR, e_oswald)
    # (3) CD_0 from the claimed glide ratio: L/D = CL/CD, and at best glide CD = CD_0 + CD_i.
    ld = max(2.0, spec["ld_claimed"])
    cd0 = max(0.010, cl_c / ld - cd_i)
    cd_c = cd0 + cd_i
    drag = 0.5 * RHO * Vc * Vc * S * cd_c
    # (4) thrust. Normally from the published climb rate with margin, so the model's ceiling
    # is above the published performance rather than exactly on it. But a published engine
    # rating is a better input than a climb rate nobody has measured, so when a spec gives
    # one (5) it becomes THRUST_MAX and the climb rate is derived by inverting the same
    # equation: T = D + W sin(gamma).
    thrust_pub = spec.get("thrust_n")
    if thrust_pub:
        thrust = thrust_pub
        excess = thrust - drag
        sin_g = excess / W
        climb = (math.inf if sin_g >= 1.0
                 else Vc * sin_g / math.sqrt(max(0.0, 1.0 - sin_g * sin_g)))
        thrust_basis = "PUBLISHED engine rating; the climb rate is what it buys"
    else:
        Vz = spec["climb_ms"]
        sin_g = Vz / math.sqrt(Vc * Vc + Vz * Vz)
        thrust = (drag + W * sin_g) * margin
        climb = Vz
        thrust_basis = "derived from the published climb rate, with margin"
    return dict(mass_kg=round(spec["mass_kg"], 3),
                wing_area_m2=round(S, 4), wing_span_m=round(b, 3),
                chord_m=round(S / b, 4),
                ixx=round(spec["mass_kg"] * (spec["length_m"] ** 2) / 60.0, 5),
                iyy=round(spec["mass_kg"] * (b ** 2) / 36.0, 4),
                izz=round(spec["mass_kg"] * (b ** 2) / 28.0, 4),
                thrust_max_n=round(thrust, 3),
                cl0=cl0, cl_alpha=cl_alpha, cl_elevator=cl_elevator,
                alpha_stall_rad=round(alpha_stall, 6),
                cd0=round(cd0, 5), cd_induced_k=round(1.0 / (math.pi * AR * e_oswald), 5),
                cm0=0.02, cm_alpha=-0.55, cm_q=-25.0, cm_elevator=0.35,
                cy_beta=-0.35, cl_beta=-0.06, cl_p=-0.45,
                cn_beta=0.055, cn_r=-0.15, cn_rudder=0.06,
                launch_speed_mps=max(spec["stall_ms"] * 1.25, 16.0),
                launch_pitch_deg=12.0, launch_alt_m=2.0,
                _derived=dict(cl_max=round(cl_max, 4), aspect_ratio=round(AR, 2),
                              cl_cruise=round(cl_c, 4), cd_cruise=round(cd_c, 5),
                              cl_alpha_per_deg=round(cl_alpha / 57.2958, 3),
                              wing_loading_kg_m2=round(spec["mass_kg"] / S, 1),
                              thrust_to_weight=round(thrust / W, 3),
                              thrust_basis=thrust_basis,
                              thrust_is_published=bool(thrust_pub),
                              climb_ms=round(climb, 2) if climb != math.inf else None,
                              cruise_drag_n=round(drag, 1),
                              thrust_margin_over_cruise_drag=round(thrust / drag, 3),
                              # Excess thrust at cruise, as a fraction of weight: the only
                              # longitudinal acceleration the aircraft has in level flight.
                              excess_thrust_frac=round((thrust - drag) / W, 4),
                              max_level_speed_ms=(
                                  round(thrust_limited_speed(W, S, cd0, 1.0 / (math.pi * AR * e_oswald), thrust), 2)
                                  if thrust_limited_speed(W, S, cd0, 1.0 / (math.pi * AR * e_oswald), thrust)
                                  else None),
                              stall_weight_thrust_n=round(W, 2)))


def cl_i_cd(cl, AR, e):
    return cl * cl / (math.pi * AR * e)


def thrust_limited_speed(W, S, cd0, k_ind, thrust):
    """Level-flight speed at which THRUST balances drag: the polar's own top speed.

    Same parabolic polar the plant flies, so
        D(V) = 0.5 rho V^2 S CD_0 + k W^2 / (0.5 rho V^2 S),      k = 1/(pi A e)
    and D = T is a quadratic in x = V^2:
        (0.5 rho S CD_0) x^2 - T x + (k W^2 / (0.5 rho S)) = 0.

    CAUTION, and it is why this is reported separately from the cruise speed: it assumes
    thrust does not fall off with speed, which is true of a turbojet and badly wrong for a
    propeller. A piston engine gives P/V, so a Shahed-136 whose 37 kW is 720 N at 51 m/s is
    down to 398 N at 93 m/s, and the constant-thrust answer (335 km/h) is nearly double
    what the aircraft does. `thrust_is_published` in the derived block says which case a
    given profile is.
    """
    A = 0.5 * RHO * S * cd0
    B = k_ind * W * W / (0.5 * RHO * S)
    disc = thrust * thrust - 4.0 * A * B
    if disc <= 0.0:
        return None                      # thrust never beats drag: not an aircraft
    x = (thrust + math.sqrt(disc)) / (2.0 * A)
    return math.sqrt(x)


# PX4's own 10015_gazebo-classic_iris geometry and inertia, used to scale every other
# multirotor profile. Anchoring to a measured airframe is deliberate: an inertia is not
# derivable from a published mass and wheelbase without knowing where the mass actually
# sits, and the obvious point-mass assumption (all the mass at the rotors) is wrong by a
# factor of ~10 because most of a real quad's mass is the battery and the motors, in the
# middle. Measured on this rig: a Mavic 3 profile built that way declares Ixx = 0.30 kg m^2
# against the Iris's 0.029, PX4's attitude controller and the plant disagree by that factor,
# and the EKF refuses to arm with "Preflight Fail: Attitude failure (roll)".
#
# So inertia is scaled from the Iris by mass and by wheelbase squared -- the two published
# numbers that actually determine it -- and rotor positions keep the Iris's proportions,
# which are a "plus" layout (rotors on the body axes, not on the diagonals).
PX4_IRIS = dict(
    mass_kg=1.5, wheelbase_m=0.795,
    rotor_x=0.1515, rotor_y=0.245, rotor_km=0.05,
    ixx=0.029125, iyy=0.029125, izz=0.055225)


def derive_quad(spec, tau=0.02):
    """simuav `QuadAirframe` coefficients: geometry from the wheelbase, inertia scaled."""
    m = spec["mass_kg"]
    k = spec["wheelbase_m"] / PX4_IRIS["wheelbase_m"]          # linear scale
    rx, ry = PX4_IRIS["rotor_x"] * k, PX4_IRIS["rotor_y"] * k
    km = PX4_IRIS["rotor_km"]
    # PX4's own rotor ordering and signs for the Iris, scaled. Getting this wrong is not a
    # cosmetic difference: the moment coefficients and the control allocation have to agree
    # with the rotor positions or the vehicle pitches the wrong way for a given command.
    rotors = ((rx, ry, km), (-rx, -ry * 0.765, km), (rx, -ry, -km), (-rx, ry * 0.765, -km))
    km_scale = m * (spec["wheelbase_m"] ** 2) / (PX4_IRIS["mass_kg"] * PX4_IRIS["wheelbase_m"] ** 2)
    return dict(mass_kg=round(m, 4),
                ixx=round(PX4_IRIS["ixx"] * km_scale, 6),
                iyy=round(PX4_IRIS["iyy"] * km_scale, 6),
                izz=round(PX4_IRIS["izz"] * km_scale, 6),
                thrust_max_n=round(spec["thrust_per_rotor_n"], 3),
                motor_tau_s=tau,
                rotors=[list(x) for x in rotors],
                drag_body=[0.25, 0.25, 0.35],
                rot_damping=0.02, drag_quad_body=[0.0, 0.0, 0.0],
                _derived=dict(scale=round(k, 4),
                              inertia_scale=round(km_scale, 5),
                              iris_reference=f"Iris Ixx {PX4_IRIS['ixx']} x "
                                            f"{km_scale:.3f}",
                              hover_thrust_n=round(m * G, 3),
                              thrust_to_weight=round(4 * spec["thrust_per_rotor_n"] / (m * G), 2)))


# ------------------------------------------------------------------- the PX4 envelope
#
# The autopilot's half of an aircraft is NOT stored in the profile and NOT written by this
# script: it is recomputed from the profile's published performance every time the profile is
# loaded, by swarmpu-simuav's engine/fdm/profiles.py:plane_envelope(), because both this repo's
# reference plant and the sim that pushes the parameters have to get the same answer from the
# same numbers.
#
# So config/px4/fixedwing.json holds only what no aircraft's performance decides -- the harness
# settings (disarm, launch detection), the throttle range, the TECS behaviours, and the
# control-loop tuning that is a property of PX4's loops (NPFG_DAMPING, FW_R_TC, FW_R_LIM).
# Everything else -- the airspeed envelope, the pitch and vertical-rate limits, the throttle
# trim, the loiter radius, the path-following period -- is arithmetic on what is below.
#
# It used to be one tuned table for every aircraft, which is the failure this split fixes:
# a 16 m/s trim and a 30 m/s ceiling applied to a 850 kg Geran-5 with 1,960 N of thrust and a
# published 125 m/s cruise would have had TECS holding it at a fifth of its speed, with a
# fifth of the lift and a turn radius a twentieth of its own.

SIMUAV_ROOT = pathlib.Path(os.environ.get("SIMUAV_ROOT", "/home/nis/dev/swarmpu-simuav"))
if str(SIMUAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMUAV_ROOT))
from engine.fdm import profiles as simuav_profiles   # noqa: E402  (the envelope implementation)


def build_plane(name, spec, force=False):
    d = derive_plane(spec, **spec.get("aero", {}))
    derived = d.pop("_derived")
    problems = consistency(spec, d, derived)
    if problems and not force:
        raise ValueError(
            f"{name}: the published figures are not mutually consistent, so no honest "
            f"profile can be derived from them:\n    - " + "\n    - ".join(problems) +
            "\n  Fix the inputs or pass --force if the profile is wanted anyway (it will "
            "then fly something that does not exist).")
    climb_input = spec.get("climb_ms")
    perf = {
        "basis": "PUBLISHED manufacturer/open-source figures, not measured on this rig",
        "mass_kg": spec["mass_kg"], "wingspan_m": spec["wing_span_m"],
        "stall_ms": spec["stall_ms"], "cruise_ms": spec["cruise_ms"],
        "climb_ms": climb_input if climb_input is not None else derived["climb_ms"],
        "climb_basis": ("published/estimated input" if climb_input is not None
                        else "DERIVED from the published engine thrust"),
        "ld_claimed": spec["ld_claimed"],
        "consistency_problems": problems or ["none"],
        "derived": derived,
    }
    # The top speed the envelope will use, reported rather than stored: profiles.plane_envelope
    # recomputes it from these same numbers every time the profile is loaded, so putting it in
    # the file as well would be a second copy to keep in step.
    v_max = simuav_profiles.plane_top_speed(perf, derived, d)
    perf["max_speed_ms"] = round(v_max, 2)
    perf["max_speed_basis"] = ("published top speed" if spec.get("max_speed_ms") else
                               ("thrust-limited on the derived polar"
                                if derived["thrust_is_published"] else
                                "cruise x 1.15 (no published top speed; a propeller's plant is "
                               "power-limited, so the polar answer would be optimistic)"))
    return {
        # A spec can lower its own confidence: a profile whose inputs are mostly estimates
        # must not be stamped "published" just because one of its numbers is.
        "name": name, "title": spec["title"], "kind": "plane",
        "confidence": spec.get("confidence", "published"),
        "sources": spec["sources"], "notes": spec["notes"],
        "launch": "catapult", "px4_sim_model": "gazebo-classic_plane",
        "performance": perf,
        "airframe": d,
    }


def build_quad(name, spec, force=False):
    d = derive_quad(spec)
    derived = d.pop("_derived")
    m, tpr = spec["mass_kg"], spec["thrust_per_rotor_n"]
    # The autopilot has to be told about the airframe too, or the two halves of the
    # simulation describe different aircraft.
    #
    # PX4's 10015_gazebo-classic_iris airframe script pins the rotor geometry
    # (CA_ROTOR0_PX .. CA_ROTOR3_KM) as `param set-default`, so a profile that changes the
    # FDM's rotor layout without also changing those leaves PX4 mixing with the Iris's
    # geometry while the plant flies something else. Measured: the FDM ran it happily at
    # real-time factor 1.0 while PX4 refused to arm with "Preflight Fail: Attitude failure
    # (roll)" and then "High Gyro Bias" -- an EKF that could not reconcile the two.
    #
    # This is why simuav's own iris.json is empty: the Iris IS the PX4 default, so the two
    # halves agree without anyone saying so. Every other multirotor has to say it out loud,
    # which is what simuav's `px4_params` hook exists for.
    px4 = {}
    for i, (px, py, km) in enumerate(d["rotors"]):
        px4[f"CA_ROTOR{i}_PX"] = round(float(px), 4)
        px4[f"CA_ROTOR{i}_PY"] = round(float(py), 4)
        px4[f"CA_ROTOR{i}_KM"] = round(float(km), 4)
    # Hover thrust as a fraction of maximum, which is what MPC_THR_HOVER means.
    px4["MPC_THR_HOVER"] = round(m * G / (spec["n_rotors"] * tpr), 4)
    problems = []
    if px4["MPC_THR_HOVER"] >= 0.95:
        problems.append(f"hover thrust {px4['MPC_THR_HOVER']:.2f} of maximum leaves no "
                        f"thrust for anything else; the published mass and thrust do not "
                        f"describe a flyable aircraft")
    if problems and not force:
        raise ValueError(f"{name}: " + "; ".join(problems))
    return {
        "name": name, "title": spec["title"], "kind": "multirotor",
        "confidence": spec.get("confidence", "published"),
        "sources": spec["sources"], "notes": spec["notes"], "launch": None,
        "px4_sim_model": "gazebo-classic_iris",
        "px4_params": px4,
        "performance": {
            "basis": "PUBLISHED manufacturer specifications, not measured on this rig",
            "mass_kg": spec["mass_kg"], "wheelbase_m": spec["wheelbase_m"],
            "max_speed_mps": spec.get("max_speed_mps"),
            "consistency_problems": problems or ["none"],
            "derived": derived,
        },
        "airframe": d,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plane", action="append", default=[])
    ap.add_argument("--quad", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="/home/nis/dev/swarmpu-simuav/config/profiles")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="load each profile back through simuav's own loader")
    args = ap.parse_args()

    if args.list or not (args.all or args.plane or args.quad):
        print("planes (published performance):")
        for n, s in PLANES.items():
            climb = s.get("climb_ms")
            climb = (f"{climb:5.1f} m/s" if climb is not None
                     else f"{s['thrust_n']:6.0f} N thrust")
            print(f"  {n:16s} {s['mass_kg']:7.1f} kg  span {s['wing_span_m']:5.2f} m  "
                  f"cruise {s['cruise_ms']:5.1f} m/s  {climb}  -- {s['title']}")
        print("multirotors:")
        for n, s in QUADS.items():
            print(f"  {n:16s} {s['mass_kg']:7.3f} kg  {s['n_rotors']} rotors  "
                  f"wheelbase {s['wheelbase_m']:.3f} m  -- {s['title']}")
        return 0

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for n in (args.plane or (list(PLANES) if args.all else [])):
        try:
            prof = build_plane(n, PLANES[n])
        except ValueError as e:
            print(f"\n=== {n} ===\n  SKIPPED: {e}")
            continue
        written.append((out / f"{n}.json", prof))
    for n in (args.quad or (list(QUADS) if args.all else [])):
        try:
            prof = build_quad(n, QUADS[n])
        except ValueError as e:
            print(f"\n=== {n} ===\n  SKIPPED: {e}")
            continue
        written.append((out / f"{n}.json", prof))

    for path, prof in written:
        perf = prof["performance"]
        der = perf["derived"]
        print(f"\n=== {prof['name']} ({prof['kind']}) ===  confidence: {prof['confidence']}")
        if prof["kind"] == "plane":
            print(f"  published  {perf['mass_kg']:.1f} kg  span {perf['wingspan_m']:.1f} m  "
                  f"stall {perf['stall_ms']:.0f} m/s  cruise {perf['cruise_ms']:.0f} m/s  "
                  f"climb {perf['climb_ms']:.1f} m/s ({perf['climb_basis'][:9]})  "
                  f"L/D {perf['ld_claimed']:.0f}")
            print(f"  derived    S {prof['airframe']['wing_area_m2']:.3f} m2  "
                  f"AR {der['aspect_ratio']:.1f}  CLmax {der['cl_max']:.2f}  "
                  f"CLcruise {der['cl_cruise']:.3f}  CDcruise {der['cd_cruise']:.4f}  "
                  f"CD0 {prof['airframe']['cd0']:.4f}  "
                  f"thrust {prof['airframe']['thrust_max_n']:.0f} N "
                  f"(T/W {der['thrust_to_weight']:.2f}, "
                  f"drag at cruise {der['cruise_drag_n']:.0f} N)")
            print(f"  wing       {der['wing_loading_kg_m2']:.0f} kg/m2  "
                  f"CLalpha {der['cl_alpha_per_deg']:.2f}/deg  "
                  f"thrust: {der['thrust_basis']}")
            # What the autopilot will be told, recomputed by the same code the sim and
            # spu-simactions both call at load time -- not written into the profile.
            env = simuav_profiles.plane_envelope(prof)
            print(f"  PX4 env    trim {env['FW_AIRSPD_TRIM']:.1f} m/s  "
                  f"stall {env['FW_AIRSPD_STALL']:.1f}  max {env['FW_AIRSPD_MAX']:.1f}  "
                  f"climb {env['FW_T_CLMB_MAX']:.1f}  sink {env['FW_T_SINK_MAX']:.1f}  "
                  f"thr_trim {env['FW_THR_TRIM']:.2f}  loiter {env['NAV_LOITER_RAD']:.0f} m  "
                  f"NPFG {env['NPFG_PERIOD']:.1f} s")
        else:
            print(f"  published  {perf['mass_kg']:.3f} kg  "
                  f"wheelbase {perf['wheelbase_m']:.3f} m")
            print(f"  derived    thrust/rotor {prof['airframe']['thrust_max_n']:.2f} N  "
                  f"hover {der['hover_thrust_n']:.2f} N  "
                  f"thrust:weight {der['thrust_to_weight']:.1f}:1")
            print(f"  inertia    Ixx {prof['airframe']['ixx']:.4f}  Izz "
                  f"{prof['airframe']['izz']:.4f} kg m2   ({der['iris_reference']})")
            print(f"  to PX4     MPC_THR_HOVER {prof['px4_params']['MPC_THR_HOVER']:.3f}  "
                  f"CA_ROTOR0 ({prof['px4_params']['CA_ROTOR0_PX']:+.4f}, "
                  f"{prof['px4_params']['CA_ROTOR0_PY']:+.4f}, "
                  f"{prof['px4_params']['CA_ROTOR0_KM']:+.3f})")
        if args.write:
            path.write_text(json.dumps(prof, indent=2) + "\n")
            print(f"  wrote {path}")

    if args.verify and args.write:
        sys.path.insert(0, "/home/nis/dev/swarmpu-simuav")
        from engine.fdm import profiles as sp
        print("\n  verifying through simuav's own loader ...")
        for path, _ in written:
            p = sp.load_profile(str(path))
            print(f"    {p['name']:16s} ok  kind={p['kind']:11s} "
                  f"model={p['px4_sim_model']}  known profiles now: "
                  f"{len(sp.list_profiles())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())