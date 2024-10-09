{-

Monologue and Tool Use

Experiment with the design of Monologue syntax, with tool use

-}

{-# LANGUAGE OverloadedStrings #-}

module T03_train_1d_hypernets_ast where

import Data.Text as Text

-- * Unit
data Unit =
  Token Int  -- ^ a Token in the vocabulary
  | Embedding [Float]  -- ^ an embedding, not in the vocabulary

-- * Address
--   Memory blocks and Apply blocks can be addressed by a Unit
type Address = Unit  -- ^ presumably a Token, eg a human readable integer

-- * Monologue
--
-- Constraints:
--   * Memory has fixed width (for now, to simplify)

data Monologue =
  Monologue [Unit]
  | Memory Address [Unit]
  | Apply Address [Unit]

data Span = Span [(Monologue, Monologue)] (Maybe Monologue) -- Span [(Monologue, Memory)] Apply
