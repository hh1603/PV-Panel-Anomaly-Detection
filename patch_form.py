path = 'scripts/web_demo.py'
with open(path, encoding='utf-8') as f:
    src = f.read()

# The JS/CSS block that's causing f-string issues - replace with clean version
old = '''          <div id="drop-zone" onclick="document.getElementById('image').click()" ondragover="event.preventDefault();this.classList.add('drag-over')" ondragleave="this.classList.remove('drag-over')" ondrop="handleDrop(event)">
            <div id="drop-hint">&#128194; 点击选择或拖拽图片到此处</div>
            <img id="preview-img" style="display:none;max-height:180px;border-radius:8px;margin-top:8px;" alt="预览">
          </div>
          <input id="image" name="image" type="file" accept="image/*" required style="display:none" onchange="if(this.files[0])showPreview(this.files[0])">
        </div>
        <button type="submit" id="submit-btn">开始检测</button>
      </form>
      <script>
      function showPreview(file){{\'{'}} var r=new FileReader();r.onload=function(e){{\'{'}} var img=document.getElementById(\'preview-img\');img.src=e.target.result;img.style.display=\'block\';document.getElementById(\'drop-hint\').textContent=\'\\u2713 \'+file.name;{{\'}\'};r.readAsDataURL(file); {{\'}\'}
      function handleDrop(event){{\'{'}} event.preventDefault();document.getElementById(\'drop-zone\').classList.remove(\'drag-over\');var f=event.dataTransfer.files[0];if(f){{\'{'}} var inp=document.getElementById(\'image\');inp.files=event.dataTransfer.files;showPreview(f);\'}\'}} {{\'}\'}
      document.querySelector(\'form\').addEventListener(\'submit\',function(){{\'{'}} var btn=document.getElementById(\'submit-btn\');btn.textContent=\'检测中\\u2026\';btn.disabled=true;{{\'}\'});
      </script>'''

new = '''          <div id="drop-zone" onclick="document.getElementById('image').click()" ondragover="event.preventDefault();this.classList.add('drag-over')" ondragleave="this.classList.remove('drag-over')" ondrop="handleDrop(event)">
            <div id="drop-hint">&#128194; 点击选择或拖拽图片到此处</div>
            <img id="preview-img" style="display:none;max-height:180px;border-radius:8px;margin-top:8px;" alt="预览">
          </div>
          <input id="image" name="image" type="file" accept="image/*" required style="display:none" onchange="if(this.files[0])showPreview(this.files[0])">
        </div>
        <button type="submit" id="submit-btn">开始检测</button>
      </form>'''

if old in src:
    src = src.replace(old, new)
    print('old pattern found and replaced')
else:
    print('pattern not found, trying simpler approach')
    # Find and replace the entire script block
    import re
    # Remove the broken script block
    src = re.sub(r'\s*<script>.*?</script>', '', src, flags=re.DOTALL, count=1)
    # Find the closing </form> and add clean script after
    script = '''
      <script>
      function showPreview(f){var r=new FileReader();r.onload=function(e){var i=document.getElementById("preview-img");i.src=e.target.result;i.style.display="block";document.getElementById("drop-hint").textContent="\u2713 "+f.name;};r.readAsDataURL(f);}
      function handleDrop(e){e.preventDefault();document.getElementById("drop-zone").classList.remove("drag-over");var f=e.dataTransfer.files[0];if(f){document.getElementById("image").files=e.dataTransfer.files;showPreview(f);}}
      document.querySelector("form").addEventListener("submit",function(){var b=document.getElementById("submit-btn");b.textContent="\u68c0\u6d4b\u4e2d\u2026";b.disabled=true;});
      </script>'''
    src = src.replace('      </form>\n      <ul class="tips-list">', '      </form>' + script + '\n      <ul class="tips-list">', 1)
    print('used fallback replacement')

with open(path, 'w', encoding='utf-8') as f:
    f.write(src)
print('Done')
