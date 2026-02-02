//! Waveform parsing (PULSE, SIN, PWL).

use spicier_devices::Waveform;

use crate::error::{Error, Result};
use crate::lexer::Token;

use super::Parser;

impl<'a> Parser<'a> {
    /// Parse PULSE(v1 v2 td tr tf pw per)
    pub(super) fn parse_pulse_waveform(&mut self, line: usize) -> Result<Waveform> {
        // Expect opening paren
        if !matches!(self.peek(), Token::LParen) {
            return Err(Error::ParseError {
                line,
                message: "expected '(' after PULSE".to_string(),
            });
        }
        self.advance();

        let v1 = self.expect_value(line)?;
        let v2 = self.expect_value(line)?;

        // Optional parameters with defaults
        let td = self.try_expect_value().unwrap_or(0.0);
        let tr = self.try_expect_value().unwrap_or(0.0);
        let tf = self.try_expect_value().unwrap_or(0.0);
        let pw = self.try_expect_value().unwrap_or(0.0);
        let per = self.try_expect_value().unwrap_or(0.0);

        // Expect closing paren
        if !matches!(self.peek(), Token::RParen) {
            return Err(Error::ParseError {
                line,
                message: "expected ')' after PULSE parameters".to_string(),
            });
        }
        self.advance();

        Ok(Waveform::pulse(v1, v2, td, tr, tf, pw, per))
    }

    /// Parse SIN(vo va freq [td [theta [phase]]])
    pub(super) fn parse_sin_waveform(&mut self, line: usize) -> Result<Waveform> {
        if !matches!(self.peek(), Token::LParen) {
            return Err(Error::ParseError {
                line,
                message: "expected '(' after SIN".to_string(),
            });
        }
        self.advance();

        let vo = self.expect_value(line)?;
        let va = self.expect_value(line)?;
        let freq = self.expect_value(line)?;

        // Optional parameters with defaults
        let td = self.try_expect_value().unwrap_or(0.0);
        let theta = self.try_expect_value().unwrap_or(0.0);
        let phase = self.try_expect_value().unwrap_or(0.0);

        if !matches!(self.peek(), Token::RParen) {
            return Err(Error::ParseError {
                line,
                message: "expected ')' after SIN parameters".to_string(),
            });
        }
        self.advance();

        Ok(Waveform::sin_full(vo, va, freq, td, theta, phase))
    }

    /// Parse PWL(t1 v1 t2 v2 ...)
    pub(super) fn parse_pwl_waveform(&mut self, line: usize) -> Result<Waveform> {
        if !matches!(self.peek(), Token::LParen) {
            return Err(Error::ParseError {
                line,
                message: "expected '(' after PWL".to_string(),
            });
        }
        self.advance();

        let mut points = Vec::new();
        while let Some(t) = self.try_expect_value() {
            let v = self.expect_value(line)?;
            points.push((t, v));
        }

        if points.is_empty() {
            return Err(Error::ParseError {
                line,
                message: "PWL requires at least one time-value pair".to_string(),
            });
        }

        if !matches!(self.peek(), Token::RParen) {
            return Err(Error::ParseError {
                line,
                message: "expected ')' after PWL parameters".to_string(),
            });
        }
        self.advance();

        Ok(Waveform::pwl(points))
    }
}
