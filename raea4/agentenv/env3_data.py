"""Third task domain (RAEA_DOMAIN=3; PLAN_d3_domain.md): pre-publication fact-checking of a reported claim.

Structurally different from domains 1 and 2:
  * a compound state: the claim has two checkable parts (the reported FIGURE and its ATTRIBUTION/source), so
    s in {0 accurate, 1 wrong_figure, 2 wrong_source, 3 both_wrong}; bit 0 of s = figure wrong, bit 1 = source wrong;
  * two tools answer only ONE yes/no sub-question each (kind "bit0" / "bit1"): no single cheap tool identifies the state;
  * two different outlets republish the same wire item (one persistent shared fault: a wrong wire item makes both
    report 'accurate'), plus a fact-check aggregation feed with its own outage (arbitrary reports).

Tool order (indices used in code, logs and tables):
  0 A  outlet A article        cheap, accurate;  wire item may be WRONG for the session -> reports 'accurate'
  1 B  outlet B article        cheap, accurate;  SAME wire item as A (shared fault)
  2 F  figure check            cheap; answers only "does the figure match the primary record?" (bit 0); never faulty
  3 S  source check            cheap; answers only "is the attribution right?" (bit 1); never faulty
  4 G  fact-check aggregator   cheap, medium;    feed may be DOWN for the session -> arbitrary status
  5 P  press-office call       cost 3, accurate; never faulty
Budget 7. Same loss values as domains 1 and 2 (60 / 20 / 8); the severe error is publishing 'accurate' when it is not.
"""
K3, BUDGET3, N_TOOLS3 = 4, 7, 6
SRC3 = ["cache", "fig", "att", "feed", "live"]   # both outlets have source "cache": they share one wire item
KIND3 = {"fig": "bit0", "att": "bit1"}
FAMILY_SPEC3 = {   # (source, accuracy range, cost) per tool; phi: wire error; phi2: aggregator outage
    # tuned 2026-11-16 so that the exact oracle uses every tool (expected calls per episode over 12 draws: outlets
    # 0.85 / 0.62, figure / source check 0.61 / 0.78, aggregator 0.09, office 0.15); the first ranges left the
    # sub-question tools and outlet B almost unused (office dominated)
    "D3": dict(tools=[("cache", (.82, .90), 1), ("cache", (.82, .90), 1), ("fig", (.85, .93), 1), ("att", (.85, .93), 1),
                      ("feed", (.70, .80), 1), ("live", (.86, .94), 3)],
               phi=(.15, .40), phi2=(.15, .30), stuck="benign"),
}
PRIOR_ALPHA3 = [5, 2, 2, 1]; PRIOR_MIN3 = 0.04
STATES3 = ["accurate", "wrong_figure", "wrong_source", "both_wrong"]


def _sc(subject, question, sev_text, wire, outlets, fig, att, agg, office, claim_noun):
    """outlets: [(name, desc), (name, desc)]; fig/att: (tool name, backend, sub-question, [match, differs]);
    agg: (tool name, backend); office: (tool name, backend)."""
    sym_vals = lambda pre: [f"{pre}_{v}" for v in ("confirmed", "figure_disputed", "source_disputed", "both_disputed")]
    tools = [dict(name=outlets[0][0], field="verdict", values=sym_vals("story"),
                  what=f"Returns how {outlets[0][1]} currently reports the {claim_noun}."),
             dict(name=outlets[1][0], field="verdict", values=sym_vals("story"),
                  what=f"Returns how {outlets[1][1]} currently reports the {claim_noun}."),
             dict(name=fig[0], field="answer", values=fig[3], kind="bit0",
                  what=f"Answers one yes/no question only: {fig[2]} ({fig[3][0]} = yes, {fig[3][1]} = no). "
                       f"It says nothing about the attribution."),
             dict(name=att[0], field="answer", values=att[3], kind="bit1",
                  what=f"Answers one yes/no question only: {att[2]} ({att[3][0]} = yes, {att[3][1]} = no). "
                       f"It says nothing about the figure."),
             dict(name=agg[0], field="rating", values=sym_vals("rated"),
                  what=f"Returns the rating of the {claim_noun} on a fact-check aggregation feed."),
             dict(name=office[0], field="statement", values=sym_vals("office"),
                  what=f"Calls {office[1]} and records their statement on the {claim_noun}.")]
    return dict(role="a newsroom fact-checking agent", arg="claim_id", subject=subject, noun="claim", question=question,
                states=list(STATES3), sev_text=sev_text,
                backends={"cache": wire, "fig": fig[1], "att": att[1], "feed": agg[1], "live": office[1]},
                tools=tools)


_SEV = "publishing the claim as \"accurate\" when its figure or its attribution is actually wrong (a correction must follow)"
SCENARIOS3 = {
    "layoffs": _sc(
        "claim-4471", "whether the claim \"Northwind Logistics will cut 2,000 jobs, its CEO said on Monday\" is accurate, "
        "has a wrong figure, a wrong source, or both", _SEV, "wire-feed-intl-desk",
        [("outlet_a_story", "the business desk of outlet A"), ("outlet_b_story", "the markets page of outlet B")],
        ("filing_lookup", "the regulatory filings database", "does the job-cut figure match the company's regulatory filing?", ["figure_matches", "figure_differs"]),
        ("transcript_search", "the earnings-call transcript archive", "did the CEO make the statement (rather than another person)?", ["source_matches", "source_differs"]),
        ("factcheck_feed", "factcheck-aggregator-eu"), ("press_office_call", "the company's press office"), "claim"),
    "trial": _sc(
        "claim-5820", "whether the claim \"A phase-3 trial of drug VX-12 cut hospital stays by 30%, the lead investigator said\" "
        "is accurate, has a wrong figure, a wrong source, or both", _SEV, "wire-health-desk",
        [("outlet_a_story", "the health section of outlet A"), ("outlet_b_story", "the science page of outlet B")],
        ("registry_lookup", "the clinical-trial results registry", "does the 30% figure match the registered result?", ["figure_matches", "figure_differs"]),
        ("statement_search", "the university press-release archive", "did the lead investigator make the statement?", ["source_matches", "source_differs"]),
        ("factcheck_feed", "factcheck-aggregator-health"), ("press_office_call", "the sponsor's media office"), "claim"),
    "turnout": _sc(
        "claim-6093", "whether the claim \"Turnout in the regional election was 61%, the electoral commission announced\" "
        "is accurate, has a wrong figure, a wrong source, or both", _SEV, "wire-politics-desk",
        [("outlet_a_story", "the politics desk of outlet A"), ("outlet_b_story", "the live blog of outlet B")],
        ("results_lookup", "the official results portal", "does the 61% figure match the published results?", ["figure_matches", "figure_differs"]),
        ("announcement_search", "the commission's announcement archive", "did the electoral commission make the announcement?", ["source_matches", "source_differs"]),
        ("factcheck_feed", "factcheck-aggregator-pol"), ("press_office_call", "the commission's press office"), "claim"),
    "quake": _sc(
        "claim-7315", "whether the claim \"The earthquake measured magnitude 6.4, the national seismology institute said\" "
        "is accurate, has a wrong figure, a wrong source, or both", _SEV, "wire-world-desk",
        [("outlet_a_story", "the world desk of outlet A"), ("outlet_b_story", "the breaking-news page of outlet B")],
        ("catalog_lookup", "the seismic event catalog", "does the magnitude match the catalog entry?", ["figure_matches", "figure_differs"]),
        ("bulletin_search", "the institute's bulletin archive", "did the national seismology institute issue the statement?", ["source_matches", "source_differs"]),
        ("factcheck_feed", "factcheck-aggregator-world"), ("press_office_call", "the institute's duty officer"), "claim"),
    "merger": _sc(
        "claim-8142", "whether the claim \"Halcyon Foods agreed to buy Brightmill for $3.2 billion, a company spokesperson said\" "
        "is accurate, has a wrong figure, a wrong source, or both", _SEV, "wire-deals-desk",
        [("outlet_a_story", "the deals column of outlet A"), ("outlet_b_story", "the business page of outlet B")],
        ("filing_lookup", "the merger filings database", "does the $3.2 billion figure match the merger filing?", ["figure_matches", "figure_differs"]),
        ("statement_search", "the company's press-release archive", "did a company spokesperson make the statement?", ["source_matches", "source_differs"]),
        ("factcheck_feed", "factcheck-aggregator-biz"), ("press_office_call", "the company's investor-relations office"), "claim"),
    "transfer": _sc(
        "claim-9067", "whether the claim \"The club paid a 45 million euro transfer fee, its sporting director confirmed\" "
        "is accurate, has a wrong figure, a wrong source, or both", _SEV, "wire-sports-desk",
        [("outlet_a_story", "the sports desk of outlet A"), ("outlet_b_story", "the transfer blog of outlet B")],
        ("ledger_lookup", "the league's transfer register", "does the 45 million euro figure match the registered fee?", ["figure_matches", "figure_differs"]),
        ("interview_search", "the club-media interview archive", "did the sporting director make the statement?", ["source_matches", "source_differs"]),
        ("factcheck_feed", "factcheck-aggregator-sport"), ("press_office_call", "the club's media office"), "claim"),
}
