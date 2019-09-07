package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Chapter;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfWriter;

public class AddChapter {
	public static void main(String[] args) {
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document,
					new FileOutputStream("C:\\添加章节.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			Chapter chapter = new Chapter("This is chapter 1", 1);// 创建与新章节对象关联的内容
			document.add(chapter);// 向文档中添加章节
			chapter = new Chapter("This is chapter 2", 2);// 创建与新章节对象关联的内容
			document.add(chapter);// 向文档中添加章节
			chapter = new Chapter("This is chapter 3", 3);// 创建与新章节对象关联的内容
			document.add(chapter);// 向文档中添加章节
			chapter = new Chapter("This is chapter 4", 4);// 创建与新章节对象关联的内容
			document.add(chapter);// 向文档中添加章节
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}

	}
}
