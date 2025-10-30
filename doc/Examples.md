# Examples of Triangulation Schemes

postcondition:

    uv run benchmark --dataset datasets/test.json --selector Postcondition --model gpt-4o --task 1_plus

off-by-one:

    uv run benchmark --dataset datasets/test.json --selector OffByOne --model gpt-4o --task 1_plus

syntactic:

    uv run benchmark --dataset datasets/test.json --selector Syntactic --model gpt-4o --task 1_plus

fwd-inv:

    uv run benchmark --dataset datasets/test.json --selector FWD_INV --model gpt-4o --task 1_plus

fwd-sinv:

    uv run benchmark --dataset datasets/test.json --selector FWD_SINV --model gpt-4o --task 3_square

fwd-sinv w.r.t. list suffix:

    uv run benchmark --dataset datasets/test.json --selector FWD_SINV --model gpt-4o --task 2_list_sum
    
fwd-inv w.r.t. list suffix with length parameter:
    
    uv run benchmark --dataset datasets/test.json --selector FWD_SINV --model gpt-4o --task 9_list_sum_with_length
    
enum-sinv:

    uv run benchmark --dataset datasets/test.json --selector ENUM_SINV --model gpt-4o --task 4_smaller_number

stream(fwd-inv):

    uv run benchmark --dataset datasets/test.json --selector FWD_INV --model gpt-4o --task 5_stream_inc

stream(fwd-inv) with argument unpacking:

    uv run benchmark --dataset datasets/test.json --selector FWD_INV --model gpt-4o --task 8_stream_plus

stream(fwd-sinv):

    uv run benchmark --dataset datasets/test.json --selector FWD_SINV --model gpt-4o --task 6_stream_square

stream(enum-sinv):

    uv run benchmark --dataset datasets/test.json --selector ENUM_SINV --model gpt-4o --task 7_stream_smaller_number


