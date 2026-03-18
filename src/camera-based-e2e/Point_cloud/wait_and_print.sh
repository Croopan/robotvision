#!/bin/bash
JOB_ID=10419202

echo "Waiting for job $JOB_ID to start..."
while true; do
    STATE=$(squeue -h -j $JOB_ID -O state 2>/dev/null | tr -d ' ')
    if [ -z "$STATE" ]; then
        echo "Job is no longer in queue. It might be done or failed."
        break
    fi
    if [ "$STATE" == "RUNNING" ]; then
        echo "Job is running!"
        break
    fi
    sleep 5
done

# Wait another moment to ensure logs are written if it just started running or finished
sleep 5
echo "--- OUT LOG ---"
cat logs/test_${JOB_ID}.out 2>/dev/null
echo "--- ERR LOG ---"
cat logs/test_${JOB_ID}.err 2>/dev/null
