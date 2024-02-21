{-# LANGUAGE ViewPatterns #-}

{-

Running Total Problem: Synthetic dataset generation

I'm preparing a synth dataset for testing the Neuralstack on a toy problem of
keeping a running tally. Haskell is so nice to reason about this in, because it
involves complex stack operations being called at the right time.

-}

module T04_stack_experiment where

import Data.List (intercalate, transpose)
import Data.List (intercalate)
import Control.Monad (replicateM)
import System.Random (randomRIO)
import Prelude hiding(Left, Right)
import Text.Printf (printf)

data Stack a =
  a :| Stack a
  | Empty
  deriving (Show)

data StackOp =
  Push Val
  | Pop
  | WildOp
  | Peek
  deriving (Show)

data Val =
    Start -- start
  | Finish -- finish
  | Wild -- wildcard
  | Left -- left tag
  | Right -- right tag
  | I Int
  --  Control signals that dictate what should be returned, and what stack operations to do.
  | SeqStart -- sequence started
  | SeqFinish -- sequence finished
  | ReturnL -- return left
  | ReturnSum -- return sum
  | ReturnR -- return right
  deriving Show


applyOp :: Stack Val -> StackOp -> (Stack Val, Maybe Val)
applyOp stack          (Push x) = (x :| stack, Nothing)
applyOp Empty          Pop      = (Empty, Nothing)
applyOp (x :| xs)      Pop      = (xs, Just x)
applyOp stack          WildOp   = (stack, Nothing)
applyOp Empty          Peek     = (Empty, Nothing)
applyOp stack@(x :| _) Peek     = (stack, Just x)

data Model = Model {
  globalStack :: Stack Val,
  workStack :: Stack Val
  }

step ::
  Val -> -- ^ Peek global stack
  Val -> -- ^ Peek work stack
  Val -> -- ^ Input value from sequence
  (StackOp, StackOp, -- PreGlobal, PreWork
   StackOp, StackOp, -- PostGlobal, PostWork
   Val) -- output

-- in a sequence
step _ _ Start  = (WildOp, WildOp, Push SeqStart, WildOp, Wild)
step _ _ Finish = (WildOp, WildOp, Push ReturnL, WildOp, Wild)
step SeqStart (I work) (I inp) = (WildOp, Pop, WildOp, Push (I $ mod (work + inp) 10), Wild)
step SeqStart Wild (I inp) = (WildOp, Pop, WildOp, Push (I inp), Wild)

-- after sequence
step ReturnL _ _ = (Pop, WildOp, Push ReturnSum, WildOp, Left)
step ReturnSum work _ = (Pop, Pop, Push ReturnR, WildOp, work)
step ReturnR _ _ = (Pop, WildOp, WildOp, WildOp, Right)

step _ _ _ = (WildOp, WildOp, WildOp, WildOp, Wild)


process :: [Val] -> [(String, String, String, String, String, String, String, String)]
process inputs = go inputs (Model Empty Empty) where
  go [] _ = []
  go (x:xs) model@(Model global work) =
    let globalPeek = peek global
        workPeek = peek work
        (preGlobalOp, preWorkOp, postGlobalOp, postWorkOp, output) = step globalPeek workPeek x
        -- apply pre ops
        global' = fst $ applyOp global preGlobalOp
        work' = fst $ applyOp work preWorkOp
        -- apply post ops
        global'' = fst $ applyOp global' postGlobalOp
        work'' = fst $ applyOp work' postWorkOp
        newModel = Model global'' work''

        vs = valToString
        so = stackOpOpString
        sv = stackOpValString

        currentOutput = (vs x, vs output,
                         -- Pre
                         so preGlobalOp, so preWorkOp,
                         -- Post
                         so postGlobalOp, sv postGlobalOp,
                         so postWorkOp, sv postWorkOp)
    in currentOutput : go xs newModel

  peek :: Stack Val -> Val
  peek Empty = Wild
  peek (v :| _) = v


--------------------------------------------------
-- * Save Random

-- Generates a random integer within a given range
randomInt :: Int -> Int -> IO Int
randomInt low high = randomRIO (low, high)

-- Generates a single random sequence of Val values following the specified template
generateRandomSequence :: Int -> IO [Val]
generateRandomSequence len = do
  intValues <- replicateM len (randomInt 0 9)
  let intVals = map I intValues
  return $ [Start] ++ intVals ++ [Finish] ++ [Wild, Wild, Wild]

-- | Converts a Val to a String for CSV output
valToString :: Val -> String
valToString Start = "S"
valToString Finish = "F"
valToString Wild = "_"
valToString (I x) = show x
valToString Left = "<" -- left tag
valToString Right = ">" -- right tag
valToString SeqStart = "SS" -- sequence started
valToString SeqFinish = "SF" -- sequence finished
valToString ReturnL = "RL" -- return left
valToString ReturnSum = "RS" -- return sum
valToString ReturnR = "RR" -- return right

-- | Converts a StackOp's action to a String for CSV output
stackOpOpString :: StackOp -> String
stackOpOpString op = case op of
  Push val -> "PSH"
  Pop -> "POP"
  WildOp -> "NOP"
  Peek -> "PEK"

-- | Converts a StackOp's value (if it's a push op) to a string for CSV
-- output. `Wild`, if not a PushOp
stackOpValString :: StackOp -> String
stackOpValString op = case op of
  Push val -> valToString val
  _ -> valToString Wild

unzip7 :: [(a, b, c, d, e, f, g)] -> ([a], [b], [c], [d], [e], [f], [g])
unzip7 = foldr (\(a,b,c,d,e,f,g) (as,bs,cs,ds,es,fs,gs) ->
  (a:as,b:bs,c:cs,d:ds,e:es,f:fs,g:gs))
  ([],[],[],[],[],[],[])

unzip8 :: [(a, b, c, d, e, f, g, h)] -> ([a], [b], [c], [d], [e], [f], [g], [h])
unzip8 = foldr (\(a,b,c,d,e,f,g,h) (as,bs,cs,ds,es,fs,gs,hs) ->
  (a:as,b:bs,c:cs,d:ds,e:es,f:fs,g:gs,h:hs))
  ([],[],[],[],[],[],[],[])

unzip9 :: [(a, b, c, d, e, f, g, h, i)] -> ([a], [b], [c], [d], [e], [f], [g], [h], [i])
unzip9 = foldr (\(a,b,c,d,e,f,g,h,i) (as,bs,cs,ds,es,fs,gs,hs,is) ->
  (a:as,b:bs,c:cs,d:ds,e:es,f:fs,g:gs,h:hs,i:is))
  ([],[],[],[],[],[],[],[],[])

unzip10 :: [(a, b, c, d, e, f, g, h, i, j)] -> ([a], [b], [c], [d], [e], [f], [g], [h], [i], [j])
unzip10 = foldr (\(a,b,c,d,e,f,g,h,i,j) (as,bs,cs,ds,es,fs,gs,hs,is,js) ->
  (a:as,b:bs,c:cs,d:ds,e:es,f:fs,g:gs,h:hs,i:is,j:js))
  ([],[],[],[],[],[],[],[],[],[])


fmtRow :: [(String, String, String, String, String, String, String, String)] -> String
fmtRow xs = intercalate "|" (intercalate " " <$> [a, b, c, d, e, f, g, h])
  where (a, b, c, d, e, f, g, h) = unzip8 xs

saveProcessedSequencesToCSV :: FilePath -> [[Val]] -> IO ()
saveProcessedSequencesToCSV filePath sequences = do
  -- let zipo is os = (\(inp, (a,b,c,d,e,f,g)) -> intercalate "|" $ map (intercalate " ") [inp, a,b,c,d,e,f,g]) <$> zip is os
  -- let processedWithInputs :: [(String, String, String, String, String, String, String)]
  --     processedWithInputs = [zipo (map valToString inputs) (unzip7 $ process inputs) | inputs <- sequences]
  let outs = fmtRow . process <$> sequences
  let csvLines = "Input|Output|PreGlobalOp|PreWorkOp|PostGlobalOp|PostGlobalVal|PostWorkOp|PostWorkVal" : outs
  writeFile filePath (unlines csvLines)


gen :: Int -> Int -> IO ()
gen nData seqLen = do
  sequences <- replicateM nData (generateRandomSequence seqLen)
  let filePath = printf "mod_sum_length_%d.csv" seqLen
  saveProcessedSequencesToCSV filePath sequences
  putStrLn $ "Saved processed sequences with inputs to " ++ filePath

main :: IO ()
main = do
  gen 1000 3
  gen 1000 5
  gen 1000 10
  gen 1000 20

--------------------------------------------------
--

-- -- | Example
-- inputs, outputs :: [Val]
-- inputs  = [Start, I 1, I 2, I 3, I 4, Finish, Wild, Wild,    Wild, Wild]
-- outputs = [Wild, Wild,   Wild,   Wild,   Wild,   Wild, Left, I 10, Right, Wild]

-- test :: IO ()
-- test = do
--   let outs = process inputs
--   let zipped = (\(inp, (a,b,c,d,e,f,g)) -> (inp, a,b,c,d,e,f,g))
--                <$> zip inputs outs
--   let header = ["Inp", "Out",
--                 "PreGOp", "PreWOp",
--                 "PostGOp", "PostG",
--                 "PostWOp", "PostW"]

--   putStrLn $ formatGrid header zipped


--------------------------------------------------
-- * Print Grid

formatGrid :: ToLOS a => [String] -> [a] -> String
formatGrid headers rows = unlines $ headerRow : separator : formattedRows
  where
    allRows = headers : toLOS rows
    formattedRows = map formatRow $ toLOS rows
    headerRow = formatRow headers
    formatRow row = intercalate " | " [padRight maxWidths i x | (x, i) <- zip row [0..]]
    maxWidths = map (maximum . map length) (transpose allRows)
    padRight widths index str = str ++ replicate (widths !! index - length str) ' '
    separator = intercalate "-+-" [replicate width '-' | width <- maxWidths]


class ToLOS a where
  toLOS :: [a] -> [[String]]

instance (Show a, Show b) => ToLOS (a, b) where
  toLOS = fmap (\(a,b) -> [show a, show b])

instance (Show a, Show b, Show c) => ToLOS (a, b, c) where
  toLOS = map (\(a, b, c) -> [show a, show b, show c])

instance (Show a, Show b, Show c, Show d) => ToLOS (a, b, c, d) where
  toLOS = map (\(a, b, c, d) -> [show a, show b, show c, show d])

instance (Show a, Show b, Show c, Show d, Show e) => ToLOS (a, b, c, d, e) where
  toLOS = map (\(a, b, c, d, e) -> [show a, show b, show c, show d, show e])

instance (Show a, Show b, Show c, Show d, Show e, Show f) => ToLOS (a, b, c, d, e, f) where
  toLOS = map (\(a, b, c, d, e, f) -> [show a, show b, show c, show d, show e, show f])

instance (Show a, Show b, Show c, Show d, Show e, Show f, Show g) => ToLOS (a, b, c, d, e, f, g) where
  toLOS = map (\(a, b, c, d, e, f, g) -> [show a, show b, show c, show d, show e, show f, show g])

instance (Show a, Show b, Show c, Show d, Show e, Show f, Show g, Show h) => ToLOS (a, b, c, d, e, f, g, h) where
  toLOS = map (\(a, b, c, d, e, f, g, h) -> [show a, show b, show c, show d, show e, show f, show g, show h])

instance (Show a, Show b, Show c, Show d, Show e, Show f, Show g, Show h, Show i) => ToLOS (a, b, c, d, e, f, g, h, i) where
  toLOS = map (\(a, b, c, d, e, f, g, h, i) -> [show a, show b, show c, show d, show e, show f, show g, show h, show i])

instance (Show a, Show b, Show c, Show d, Show e, Show f, Show g, Show h, Show i, Show j) => ToLOS (a, b, c, d, e, f, g, h, i, j) where
  toLOS = map (\(a, b, c, d, e, f, g, h, i, j) -> [show a, show b, show c, show d, show e, show f, show g, show h, show i, show j])

instance (Show a, Show b, Show c, Show d, Show e, Show f, Show g, Show h, Show i, Show j, Show k) => ToLOS (a, b, c, d, e, f, g, h, i, j, k) where
  toLOS = map (\(a, b, c, d, e, f, g, h, i, j, k) -> [show a, show b, show c, show d, show e, show f, show g, show h, show i, show j, show k])

instance (Show a, Show b, Show c, Show d, Show e, Show f, Show g, Show h, Show i, Show j, Show k, Show l) => ToLOS (a, b, c, d, e, f, g, h, i, j, k, l) where
  toLOS = map (\(a, b, c, d, e, f, g, h, i, j, k, l) -> [show a, show b, show c, show d, show e, show f, show g, show h, show i, show j, show k, show l])

instance (Show a, Show b, Show c, Show d, Show e, Show f, Show g, Show h, Show i, Show j, Show k, Show l, Show m) => ToLOS (a, b, c, d, e, f, g, h, i, j, k, l, m) where
  toLOS = map (\(a, b, c, d, e, f, g, h, i, j, k, l, m) -> [show a, show b, show c, show d, show e, show f, show g, show h, show i, show j, show k, show l, show m])
