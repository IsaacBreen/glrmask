use std::env;

use mask_game::candidate::{
    BaselineCandidate, ComplementCandidate, CopyFirstGroupRunCandidate, GlrMaskFinalDenseCandidate,
    GlrMaskFinalDenseComplementCandidate, GlrMaskLikeCandidate, ParallelComplementCandidate,
};
use mask_game::{evaluate, load_game_data};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let data_path = args
        .next()
        .unwrap_or_else(|| "mask_game/data/example_slow_mask_game.json.gz".to_string());
    let repetitions = args
        .next()
        .map(|raw| raw.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    let candidate = args.next().unwrap_or_else(|| "complement".to_string());
    let case_filter = args.next();

    let mut data = load_game_data(&data_path)?;
    if let Some(raw_filter) = case_filter {
        data = filter_data(data, &raw_filter)?;
    }
    match candidate.as_str() {
        "baseline" => print_summary(&data, evaluate::<BaselineCandidate>(&data, repetitions)?)?,
        "group" | "glrmask_like" => {
            print_summary(&data, evaluate::<GlrMaskLikeCandidate>(&data, repetitions)?)?
        }
        "copy" | "copy_first" => print_summary(
            &data,
            evaluate::<CopyFirstGroupRunCandidate>(&data, repetitions)?,
        )?,
        "complement" => print_summary(&data, evaluate::<ComplementCandidate>(&data, repetitions)?)?,
        "parallel" | "parallel_complement" => print_summary(
            &data,
            evaluate::<ParallelComplementCandidate>(&data, repetitions)?,
        )?,
        "final" | "glrmask_final" | "glrmask_final_dense" => print_summary(
            &data,
            evaluate::<GlrMaskFinalDenseCandidate>(&data, repetitions)?,
        )?,
        "final_complement" | "glrmask_final_dense_force_complement" => print_summary(
            &data,
            evaluate::<GlrMaskFinalDenseComplementCandidate>(&data, repetitions)?,
        )?,
        "all" => {
            print_summary(&data, evaluate::<BaselineCandidate>(&data, repetitions)?)?;
            print_summary(&data, evaluate::<GlrMaskLikeCandidate>(&data, repetitions)?)?;
            print_summary(
                &data,
                evaluate::<CopyFirstGroupRunCandidate>(&data, repetitions)?,
            )?;
            print_summary(&data, evaluate::<ComplementCandidate>(&data, repetitions)?)?;
            print_summary(
                &data,
                evaluate::<ParallelComplementCandidate>(&data, repetitions)?,
            )?;
            print_summary(
                &data,
                evaluate::<GlrMaskFinalDenseCandidate>(&data, repetitions)?,
            )?;
            print_summary(
                &data,
                evaluate::<GlrMaskFinalDenseComplementCandidate>(&data, repetitions)?,
            )?;
        }
        other => {
            return Err(format!(
                "unknown candidate {other:?}; use baseline, group, copy, complement, parallel, final, or all"
            )
            .into());
        }
    }

    Ok(())
}

fn filter_data(
    mut data: mask_game::GameData,
    raw: &str,
) -> Result<mask_game::GameData, Box<dyn std::error::Error>> {
    let (problem, example_index, step) = parse_case_filter(raw)?;
    data.cases.retain(|case| {
        case.problem == problem && case.example_index == example_index && case.step == step
    });
    if data.cases.is_empty() {
        return Err(format!("case filter {raw:?} matched no cases").into());
    }
    Ok(data)
}

fn parse_case_filter(raw: &str) -> Result<(String, u32, u32), Box<dyn std::error::Error>> {
    let (problem, example, step) = raw
        .rsplit_once(':')
        .and_then(|(prefix, step)| {
            prefix
                .rsplit_once(':')
                .map(|(problem, example)| (problem, example, step))
        })
        .ok_or_else(|| format!("case filter must be PROBLEM:EXAMPLE:STEP, got {raw:?}"))?;
    Ok((problem.to_string(), example.parse()?, step.parse()?))
}

fn print_summary(
    data: &mask_game::GameData,
    summary: mask_game::EvalSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("candidate: {}", summary.candidate);
    println!("source: {}", data.source);
    println!("maps: {}", data.maps.len());
    println!("cases: {}", summary.cases);
    println!("repetitions: {}", summary.repetitions);
    println!("calls: {}", summary.total_calls);
    println!("mean_ns: {:.1}", summary.mean_ns);
    println!("p50_ns: {}", summary.p50_ns);
    println!("p95_ns: {}", summary.p95_ns);
    println!("p99_ns: {}", summary.p99_ns);
    println!("max_ns: {}", summary.max_ns);
    println!("stabilized_max_ns: {}", summary.stabilized_max_ns);
    if let Some(case) = data.cases.get(summary.max_case_index) {
        println!("max_case_index: {}", summary.max_case_index);
        println!("max_repetition: {}", summary.max_repetition);
        println!("max_problem: {}", case.problem);
        println!("max_example_index: {}", case.example_index);
        println!("max_step: {}", case.step);
        if let Some(token_id) = case.token_id {
            println!("max_token_id: {}", token_id);
        }
        if let Some(allowed_count) = case.allowed_count {
            println!("max_allowed_count: {}", allowed_count);
        }
        println!("max_internal_ids: {}", case.internal_ids.len());
        println!("max_expected_words: {}", case.expected_sparse_words.len());
    }
    if let Some(case) = data.cases.get(summary.stabilized_max_case_index) {
        println!("stabilized_max_case_index: {}", summary.stabilized_max_case_index);
        println!("stabilized_max_problem: {}", case.problem);
        println!("stabilized_max_example_index: {}", case.example_index);
        println!("stabilized_max_step: {}", case.step);
        if let Some(token_id) = case.token_id {
            println!("stabilized_max_token_id: {}", token_id);
        }
        if let Some(allowed_count) = case.allowed_count {
            println!("stabilized_max_allowed_count: {}", allowed_count);
        }
        println!("stabilized_max_internal_ids: {}", case.internal_ids.len());
        println!("stabilized_max_expected_words: {}", case.expected_sparse_words.len());
    }
    println!();

    Ok(())
}
