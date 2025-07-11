# 🎯 COMPLETE GitHub Setup for Beginners

## FIRST: Install Git

### Option 1: Download Git
1. Go to: https://git-scm.com/download/win
2. Download "64-bit Git for Windows Setup"
3. Run the installer
4. **Use all default settings** - just keep clicking "Next"
5. Restart your PowerShell after installation

### Option 2: Install with Winget (if you have it)
```powershell
winget install --id Git.Git -e --source winget
```

---

## SECOND: Create GitHub Account
1. Go to https://github.com
2. Click "Sign up" 
3. Choose username (use something professional like: firstname-lastname-dev)
4. Verify your email

---

## THIRD: Create Repository
1. In GitHub, click green **"New"** button
2. Repository name: `flight-dynamics-simulation`
3. Description: `Professional 6-DOF aircraft flight simulation in Python`
4. Make it **PUBLIC** ✅
5. **Do NOT** check "Add README" ❌
6. Click **"Create repository"**

---

## FOURTH: Upload Your Code

**Open PowerShell in your flight-dynamics folder and run:**

```powershell
# Step 1: Initialize Git
git init

# Step 2: Configure Git (use your info!)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Step 3: Add all files
git add .

# Step 4: Make first commit
git commit -m "Initial commit: Professional flight dynamics simulation"

# Step 5: Set main branch
git branch -M main

# Step 6: Connect to GitHub (REPLACE WITH YOUR INFO!)
git remote add origin https://github.com/YOUR_USERNAME/flight-dynamics-simulation.git

# Step 7: Upload everything
git push -u origin main
```

**REPLACE:**
- `YOUR_USERNAME` with your GitHub username
- `Your Name` with your real name
- `your.email@example.com` with your email

---

## 🚨 If You Get Errors

**"git: command not found"**
- Install Git first (see above)
- Restart PowerShell

**"Permission denied"**
- Make sure you're logged into GitHub in your browser
- Double-check your username in the URL

**"Repository not found"**
- Make sure the repository name matches exactly
- Check that you created it as PUBLIC

---

## ✅ Success Checklist

After running the commands:
- [ ] Go to `https://github.com/YOUR_USERNAME/flight-dynamics-simulation`
- [ ] See all your Python files listed
- [ ] See your README.md displaying nicely
- [ ] Star your own repository (click the star ⭐)

---

## 🎉 You're Done!

**Your project is now on GitHub!** 

**Share this URL on your resume:**
`https://github.com/YOUR_USERNAME/flight-dynamics-simulation`

This shows employers you can:
- Build complex engineering simulations
- Use version control (Git/GitHub)  
- Write clean, professional code
- Document your work properly

**This is impressive work that will get you noticed!** 🚀
