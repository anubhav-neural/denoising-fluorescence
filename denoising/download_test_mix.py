import gdown

url = "https://drive.google.com/uc?id=13bAS6Ipuk2xZkF99FTdanJ3fenvA3QDj"

output = "test_mix.tar"

print("Downloading test_mix archive...")
gdown.download(url, output, quiet=False)
print("Download complete.")