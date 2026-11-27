//! Flash-Next configuration, validated once before model allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub proj: bool,
    pub frontend: bool,
    pub fp4_ep: bool,
}
impl Options {
    pub const NAMES: [&str; 3] = [
        "VADE_FLASHNEXT_FAST_PROJ",
        "VADE_FLASHNEXT_FAST_FRONTEND",
        "VADE_FLASHNEXT_FP4_EP2",
    ];
    pub fn parse(
        mut value: impl FnMut(&str) -> Result<Option<String>, String>,
    ) -> Result<Self, String> {
        let mut flags = [false; 3];
        for (flag, name) in flags.iter_mut().zip(Self::NAMES) {
            *flag = match value(name)?.as_deref() {
                None | Some("0") => false,
                Some("1") => true,
                _ => return Err(format!("{name} must be 0 or 1")),
            };
        }
        let [proj, frontend, fp4_ep] = flags;
        Ok(Self {
            proj,
            frontend,
            fp4_ep,
        })
    }
    pub fn load() -> Result<Self, String> {
        Self::parse(|name| match std::env::var(name) {
            Ok(v) => Ok(Some(v)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(_) => Err(format!("{name} must be 0 or 1")),
        })
    }
    #[cfg(any(target_os = "linux", test))]
    pub(crate) fn validate_rows(self) -> Result<(), String> {
        if self.proj || self.frontend {
            return Err("batched rows require FAST_PROJ and FAST_FRONTEND OFF".into());
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::Options;
    #[test]
    fn strict_load_time_configuration() {
        assert_eq!(Options::parse(|_| Ok(None)).unwrap(), Options::default());
        for mask in 0..8 {
            let mut reads = 0;
            let options = Options::parse(|_| {
                let v = (mask >> reads) & 1;
                reads += 1;
                Ok(Some(v.to_string()))
            })
            .unwrap();
            assert_eq!(reads, 3);
            assert_eq!(
                [options.proj, options.frontend, options.fp4_ep],
                std::array::from_fn(|i| mask & (1 << i) != 0)
            );
        }
        for bad in Options::NAMES {
            for value in ["", "yes", "true", "2", " 1"] {
                assert!(
                    Options::parse(|n| Ok(Some(if n == bad { value } else { "0" }.into())))
                        .is_err()
                );
            }
            assert!(
                Options::parse(|n| if n == bad {
                    Err("non-Unicode".into())
                } else {
                    Ok(None)
                })
                .is_err()
            );
        }
    }
    #[test]
    fn rows_preserve_baseline_admission() {
        assert!(Options::default().validate_rows().is_ok());
        let rows = Options::default();
        assert!(rows.validate_rows().is_ok());
        assert!(Options { proj: true, ..rows }.validate_rows().is_err());
        assert!(
            Options {
                frontend: true,
                ..rows
            }
            .validate_rows()
            .is_err()
        );
    }
}
