use glrmask_runtime::Constraint;

fn usage() -> ! {
    eprintln!(
        "Usage: runtime_state --constraint <constraint.bin> [--state-in <state.bin>] [--state-out <state.bin>] [--tokens-json <tokens.json>] [--commit-token <id>]... [--mask-json]"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut constraint_path: Option<String> = None;
    let mut state_in_path: Option<String> = None;
    let mut state_out_path: Option<String> = None;
    let mut tokens_json_path: Option<String> = None;
    let mut commit_tokens: Vec<u32> = Vec::new();
    let mut print_mask_json = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--constraint" => {
                i += 1;
                if i >= args.len() {
                    usage();
                }
                constraint_path = Some(args[i].clone());
            }
            "--state-in" => {
                i += 1;
                if i >= args.len() {
                    usage();
                }
                state_in_path = Some(args[i].clone());
            }
            "--state-out" => {
                i += 1;
                if i >= args.len() {
                    usage();
                }
                state_out_path = Some(args[i].clone());
            }
            "--tokens-json" => {
                i += 1;
                if i >= args.len() {
                    usage();
                }
                tokens_json_path = Some(args[i].clone());
            }
            "--commit-token" => {
                i += 1;
                if i >= args.len() {
                    usage();
                }
                commit_tokens.push(args[i].parse()?);
            }
            "--mask-json" => {
                print_mask_json = true;
            }
            _ => usage(),
        }
        i += 1;
    }

    let constraint_path = constraint_path.unwrap_or_else(|| usage());
    let constraint_bytes = std::fs::read(&constraint_path)?;
    let constraint = Constraint::load(&constraint_bytes)?;

    let mut state = if let Some(path) = state_in_path {
        let state_bytes = std::fs::read(path)?;
        constraint.load_state(&state_bytes)?
    } else {
        constraint.start()
    };

    if let Some(path) = tokens_json_path {
        let json = std::fs::read_to_string(path)?;
        let tokens: Vec<u32> = serde_json::from_str(&json)?;
        state.commit_tokens(&tokens).map_err(std::io::Error::other)?;
    }

    if !commit_tokens.is_empty() {
        state
            .commit_tokens(&commit_tokens)
            .map_err(std::io::Error::other)?;
    }

    let mask = state.mask();
    let allowed_token_count: usize = mask.iter().map(|word| word.count_ones() as usize).sum();

    println!("is_finished: {}", state.is_finished());
    println!("mask_len_words: {}", mask.len());
    println!("allowed_token_count: {}", allowed_token_count);
    println!("state_summary: {:?}", state.summary());

    if print_mask_json {
        println!("mask_json: {}", serde_json::to_string(&mask)?);
    }

    if let Some(path) = state_out_path {
        std::fs::write(path, state.save())?;
    }

    Ok(())
}
