read -p "Are you sure? [Yy]" -n 1 -r
echo # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo 'Deleting experiment directories.'

  rm -rf tboard_logs/
  rm -rf runs/
  rm -rf experiments/
else
  echo 'Operation cancelled.'
fi
